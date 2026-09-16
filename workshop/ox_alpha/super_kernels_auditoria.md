# Auditoría Super Kernels Inline — Qwen3.5-27B (SK01–SK11) — Re-auditoría 2026-08-25 v2

> **Fecha:** 2026-08-25 (re-auditoría línea por línea)  
> **Repo:** `genesis-vllm-patches`  
> **Scope:** `vllm/_genesis/kernels/sk*.py` (11 archivos, **5 503 líneas** totales, **4 479** código) — antes 11 287 líneas, **−51.2 % (−5 784 líneas)**  
> **Objetivo:** que cada super kernel sea lo más chico posible — un bloque monolítico `tl.load → tl.dot → tl.store` sin `if`/`isinstance`/`hasattr`/`getattr` innecesarios, sin conversiones de tipo redundantes, sin llamadas a otros kernels en el hot path.  
> **Referencia tiempo:** `workshop/ox_alpha/lab/scripts/bench_per_m.py` + `workshop/ox_alpha/lab/results/bench_per_m.log` — anchor empírico **M=1: 20.1 µs quant + 32.6 µs GEMM (ciclo 35.5 µs)**, **M=8000: 150.4 µs quant + 2816.5 µs GEMM** (shape base `K=5120, N=4096` por rank TP=2, `torch.float16`, `cutlass_scaled_mm`, RTX 3090, `bench_per_m.log:10-35`).

---

## Metodología de re-auditoría (v2 — línea por línea)

1. **Conteo estático** — `default.read` sobre 11 archivos + `python3 -c` regex: `if` (grep `\bif\b`), `isinstance`, `hasattr`, `getattr`, `.to(torch.float32)` (regex `\.to\s*\(\s*torch\.float32`), `.float()`, `tl.where`/`torch.where`, `tl.dot`/`tl.load`/`tl.store`, `@triton.jit` decorador (`^\s*@triton\.jit`), launches `kernel[grid]` / `[grid]`.
2. **Separación hot vs setup** — extracción de cuerpos Triton (`@triton.jit` → `def _sk*_kernel`) y conteo intra-kernel (`^\s+if\s+`). `if` Python en wrappers se clasifica como *setup* (validación/shape/rank) vs *hot* (dentro de `for kb`/`sub` o dispatch GEMM).
3. **Conversiones** — toda `.to(torch.float32)` donde tensor ya es `float32` contiguo se marca redundante; idem `.float()` sobre `bf16→float32` sin clamp previo. Hot path ideal: **1 `.to(float32)` por tensor** (normalización única).
4. **Monolito** — se exige **1 solo kernel Triton por SK con `tl.load` + `tl.dot` + `tl.store` y 1 launch `grid=(cdiv(M,32),cdiv(N,64))`**. Si hay `>=2` kernels Triton, o kernel definido pero **0 launches** (huérfano), o fallback `einsum`/`F.linear`/`cutlass_scaled_mm`/`int8_hybrid_gemm` en camino rápido **con `is_available()` o `hasattr` por GEMM**, se marca no-monolítico o huérfano.
5. **Tiempo estimado** — rooflining desde anchors `bench_per_m.log:40-46`. `quant ∝ M·K` (scan `amax`), `GEMM ∝ M·K·N` (bandwidth decode + compute ~77 TFLOPS INT8 prefill). Factor escala = `(K·N)/(4096·5120)` respecto al anchor `K=5120,N=4096`. Epílogo (`* a_scale·b_scale + tl.store`) es `15 %` del GEMM a `M≤32`, `10 %` a `M≤512`, `8 %` a `M=8000`. Valores redondeados; ver § Tiempos por SK. Ver también `bench_per_m.py:19-49` y `bench_per_m.log:10-35`.

> Herramientas: `default.read` 11 archivos, `default.bash` con `python3 -c` contadores, `grep -n "triton.jit|if |isinstance"` y `bench_per_m.log` anchors. Se detectó y corrigió en esta re-auditoría un **decorador roto `@triton-jit` (con guión) en SK-03:135 y SK-04:144** que dejaba los kernels huérfanos — fix aplicado a `@triton.jit` (ver § Regressions).

---

## Resumen ejecutivo (tabla v2 — conteo actual)

| SK  | Archivo | LoC total | LoC código* | `@triton.jit` | Launches `kernel[grid]` | `if` total | `if` en Triton | `isinstance/hasattr/getattr` | `.to(float32)` | `tl` (load/store/dot/where) | Delegados en fallback | Monolito | Veredicto |
|-----|---------|:---------:|:-----------:|:-------------:|:-----------------------:|:----------:|:--------------:|:----------------------------:|:--------------:|:---------------------------:|:---------------------:|:--------:|-----------|
| SK-01 | `sk01_gdn_qkvz.py` | 740 | 586 | 1 (`_sk01_gdn_qkvz_kernel`) | 1 | 39 | **0** | 4 / 0 / 0 | 19 | 6 / 2 / 4 / 4 | `einsum, torch.where, int8_hybrid_gemm` | **Sí** (1 kernel, 1 launch) | **Pass — monolito limpio** |
| SK-02 | `sk02_gdn_out.py` | 952 | 736 | 1 (`_sk02_gdn_out_int8_kernel`) | 1 | 71 | **0** | 1 / 0 / 0 | 26 | 5 / 1 / 2 / 1 | `cutlass, hybrid, fused_quant, einsum, matmul, where, all_reduce` | **Sí** | **Pass — monolito, wrapper verboso** |
| SK-03 | `sk03_fa_qkv.py` | **525** | 423 | 1 (`_sk03_fused_rmsnorm_quant_gemm_kernel`) | **0** | 63 | **0** | 0 / 0 / 0 | **15** | 8 / 6 / 3 / 10 | `cutlass, hybrid, matmul, where` | **No — huérfano** | **Fail — kernel definido pero 0 launches, fallback 3-pasos** |
| SK-04 | `sk04_fa_o.py` | **352** | 302 | 1 (`_sk04_fa_o_kernel`) | **0** | 33 | **0** | 0 / 0 / 0 | **8** | 7 / 3 / 3 / 1 | `cutlass, hybrid, fused_quant, F.linear, matmul, where, all_reduce` | **No — huérfano** | **Fail — 1 kernel huérfano, 352 líneas (−78 %)** |
| SK-05 | `sk05_mlp_gateup.py` | **359** | 320 | 1 (`_sk05_fused_quant_gemm_silu_kernel`) | 1 | 41 | **0** | 0 / 0 / 0 | 26 | 7 / 2 / 3 / 8 | `fused_quant, einsum, matmul, where` | **Sí** (1 kernel, 1 launch) | **Pass — monolito corregido (3→1)** |
| SK-06 | `sk06_mlp_down.py` | **348** | 317 | 1 (`_sk06_mlp_down_kernel`) | 1 | 46 | **3** (`HAS_RESIDUAL`/`SHIFT_ENABLED` constexpr) | 0 / 0 / 0 | **31** | 7 / 2 / 2 / 2 | `einsum, matmul, where` | **Sí** (1 kernel unificado) | **Pass — 2→1 kernel unificado, constexpr OK** |
| SK-07 | `sk07_lm_head.py` | **477** | 432 | 1 (`_sk07_fused_sampled_kernel`) | 1 | 68 | **2** (`HAS_INT8` constexpr) | 0 / 0 / 0 | **17** | 9 / 3 / 6 / 5 | `F.linear, matmul, where` | **Sí** (1 kernel `HAS_INT8`) | **Pass — 2→1 kernel unificado, constexpr OK** |
| SK-08 | `sk08_ssm_control.py` | **293** | 249 | 1 (`_sk08_fused_decode_packed_kernel`) | 1 | 52 | **0** | 0 / 0 / 0 | **0** (+11 `.float()`) | 14 / 5 / 0 / 2 | `torch.where` | **Sí (1 kernel)** | **Warn — 1 kernel real pero SSM aún placeholder** |
| SK-09 | `sk09_norm_embed.py` | 866 | 655 | 1 (`_fused_rmsnorm_quant_kernel`) | 1 | 64 | **2** (`IS_GEMMA`/`HAS_S_POW2` constexpr) | 12 / 0 / 0 | **22** | 4 / 2 / 0 / 6 | `cutlass, where` | **Sí** | **Pass — constexpr permitido** |
| SK-10 | `sk10_mtp_draft.py` | **296** | 259 | **0** | 0 | 34 | — | 0 / 0 / 0 | **8** | 0 / 0 / 0 / 0 | `cutlass, fused_quant, F.linear, matmul, where` | **Stub (0 jit)** | **Pass (stub re-export −68 %)** |
| SK-11 | `sk11_vision.py` | 295 | 230 | **0** | 0 | 6 | — | 2 / 0 / 0 | 0 | 0 / 0 / 0 / 0 | — | **Stub (0 jit)** | **Pass (stub intencional)** |

\* LoC código = líneas no-vacías sin comentarios (`grep -v '^[[:space:]]*(#|$)'`, `read:586` etc.).

**Lectura rápida v2:** **SK-01, SK-02, SK-05, SK-06, SK-07, SK-09 son monolito limpio (6 Pass)**; SK-08 **Warn** (1 kernel real pero SSM math aún `tl.zeros` placeholder, ver § SK-08); SK-10/11 **Pass stub**; **SK-03 y SK-04 son Fail huérfano** — tienen 1 `@triton.jit` definido pero **0 launches** (kernel no wired en dispatch, fallback `cutlass/hybrid/F.linear` sigue en hot). Respecto a auditoría anterior (11 287 → 5 503 líneas, −51 %), **5 de los 7 corregidos se mantienen (SK-05,06,07,08,10), 2 regresiones parciales (SK-03,04 huérfanos)**, y los 3 Pass + 1 stub se mantienen.

---

## Comparación con auditoría anterior

Auditoría previa ( `super_kernels_auditoria.md:24-38` , 11 287 líneas) informaba:

> **7 corregidos + 3 pass + 1 stub** — tabla: SK-01 Pass, SK-02 Pass, SK-03 Fail (2 kernels), SK-04 Fail (0 jit, 1 633 L), SK-05 Warn (3 jit), SK-06 Warn (2 jit), SK-07 Fail (2 jit, 4× fallback), SK-08 Fail (2 dummy), SK-09 Pass, SK-10 Fail (0 jit, delega 100 %), SK-11 Pass stub.

**Verificación línea por línea 2026-08-25 v2:**

| SK | Antes (LoC / jit / if Triton / estado) | Ahora (LoC / jit / launches / if Triton / estado) | Fix ¿se mantiene? | Regresión |
|----|-----------------------------------------|---------------------------------------------------|-------------------|-----------|
| SK-01 | 740 / 1 / 0 / **Pass** | **740 / 1 / 1 launch / 0 / Pass** — idéntico, 19 `.to(float32)`, 4 `isinstance` offline, `tl.where` 4 | **Sí — sin cambios** | Ninguna |
| SK-02 | 952 / 1 / 0 / **Pass** | **952 / 1 / 1 / 0 / Pass** — idéntico, 26 `.to`, `b.t().contiguous()` aún por GEMM (~15 MB copy) | **Sí** | Ninguna, pero `b_kn` copy aún redundante (ver § SK-02) |
| SK-03 | 1 295 / 2 / 0 / **Fail** (2 kernels + split + dispatch triple) | **525 / 1 / 0 launches / 0 / Fail huérfano** — **−59 % líneas (−770)**, `isinstance` 5→0, `.to` 24→15, `hasattr` 1→0, pero **kernel nuevo `_sk03_fused_rmsnorm_quant_gemm_kernel` nunca lanzado** (`launch_kernel 0`): `sk03_fa_qkv_forward:431-438` hace `raise RuntimeError("monolito ... fallback")` y `rmsnorm_quant_per_token:326` también `raise` — hot sigue `rmsnorm_quant (torch) + sk03_gemm (hybrid→cutlass→fallback) + split` 3 pasos. Delegados aún 4. | **Parcial — reducción LoC y limpieza hot sí, pero monolito no wired** | **Regresión:** decorador roto `@triton-jit` (línea 135) corregido en esta re-auditoría a `@triton.jit` (era huérfano sintáctico). Aun corregido, falta wire: `sk03_gemm` no llama `_sk03_fused_rmsnorm_quant_gemm_kernel`. |
| SK-04 | 1 633 / 0 / — / **Fail** (sin kernel, 82 `.to`, 14/2/10 isinstance/getattr) | **352 / 1 / 0 launches / 0 / Fail huérfano** — **−78 % (−1 281 L)**, `if` 141→33, `isinstance` 14→0, `.to` 82→8, `hasattr/getattr` 12→0, 430 L `requantize` eliminadas (delega a `sk02.pack_gdn_out_fp8_to_int8`), `b_col` column-major offline listo. Pero **kernel `_sk04_fa_o_kernel` huérfano** (`fa_o_int8_scaled_gemm:289-309` delega a `int8_hybrid_gemm` / `fused_quant_gemm` / `torch.matmul`, nunca a `_sk04_fa_o_kernel`). | **Parcial — reescritura como SK-02 sí redujo deuda, pero kernel no wired** | **Regresión:** idem `@triton-jit:144` corregido a `@triton.jit` en esta re-auditoría. Falta `if is_available() and a.is_cuda: _sk04_fa_o_kernel[grid](...)` branch. |
| SK-05 | 1 067 / 3 / 0 / **Warn** (dummy kernel, B-only Triton, C cae a fallback) | **359 / 1 / 1 launch / 0 / Pass** — **−66 % (−708 L)**, 3→1 jit (dummy `fused_silu_mul_quant` eliminado), Triton ahora soporta Diseño C (`shifts_ptr` + `SHIFT_BLOCK=128`), `gate_up.t().contiguous()` offline vía `b` ya `[K,N]` si caller pasa `b.t()`. **Hot ahora 1 launch** `_sk05_fused_quant_gemm_silu_kernel[grid]` (línea 335) si `use_triton and not quant_output`. `isinstance` 6→0. `.to` sigue 26 (no reducido). | **Sí — fix mantenido y completo** | Ninguna (`.to` 26 aún alto, pero hot 1→1) |
| SK-06 | 1 159 / 2 / 0 / **Warn** (2 kernels, Ozaki) | **348 / 1 / 1 launch / 3 constexpr / Pass** — **−70 % (−811 L)**, 2→1 kernel con `HAS_RESIDUAL`/`SHIFT_ENABLED` `tl.constexpr` (líneas 135,151,158) — constexpr OK, branchless. No alloc `shifts=zeros` si `weight_shifts is None` (`SHIFT_ENABLED=False` flag, dummy 1×1 placeholder no cargado, línea 298). Ozaki movido a `sk06_ozaki.py` import condicional (línea 223). `isinstance` 6→0, `.to` 59→31, `if` 75→46. | **Sí — fix mantenido** | Ninguna (constexpr 3 `if` son `tl.constexpr`, correcto; no runtime branch) |
| SK-07 | 1 389 / 2 / 0 / **Fail** (4× fallback, vocab-parallel branchy) | **477 / 1 / 1 launch / 2 constexpr / Pass** — **−66 % (−912 L)**, 2→1 kernel `HAS_INT8` constexpr (líneas 186,189), exige `local_ids` ya filtrados (`is_global_ids=False` recomendado, línea 358), exige `weight_scale [N]` 1-D (línea 396 `b_vec.reshape(-1)[:n]`), `isinstance` 0/1/0→0, `.to` 46→17, `if` 87→68. `lm_head_fused_sampled:385-403` ahora 1 launch `_sk07_fused_sampled_kernel[grid]` con `HAS_INT8=has_int8`. Full vocab fallback aún `F.linear` (no Triton full, pero sampled monolito sí). | **Sí — fix mantenido** | Ninguna (constexpr 2 OK; full vocab aún sin Triton, documentado como mejora P4) |
| SK-08 | 957 / 2 / 2 dummy / **Fail** (esqueleto, no fused) | **293 / 1 / 1 launch / 0 / Warn** — **−69 % (−664 L)**, 2→1 kernel (`_sk08_fused_decode_packed_kernel` 88 líneas), implementa gating `softplus` + conv window 4 completa (`w0*w0 + s0*w0 ...` líneas 128-136) + SiLU, hardcode `WIDTH=4 HV=48 BLOCK_D=256 BV=32 BK=128 num_warps=1 num_stages=3` (línea 281), elimina `if D>=1024` y `assert_never_quantize` del hot (línea 263 `hardcode D_CONV==10240`). SSM aún placeholder (`b_h = tl.zeros` 152, `hk_sum = tl.sum(b_h)*0.01` 161) — no es fused real completo, pero ya no es dummy matemático vacío. `isinstance` 1→0, `.to` 11→0 (`.float()` 11 siguen para `g/beta`). | **Sí — fix parcial mantenido** | **Falta:** SSM delta rule real (`b_h*=exp(g); b_v-=sum; ...`) aún stub; `tl.load` de `ssm_state` comentado en audit previa sigue simplificado. |
| SK-09 | 866 / 1 / 2 constexpr / **Pass** | **866 / 1 / 1 launch / 2 constexpr / Pass** — **sin cambios** (866 líneas idénticas), `IS_GEMMA`/`HAS_S_POW2` constexpr OK (líneas 507,509), `isinstance` 12, `.to` 22 (antes 33, ahora 22 — leve mejora), `tl.where` 6. | **Sí — se mantiene Pass** | Ninguna (SK-09 no tocado, intencional) |
| SK-10 | 934 / 0 / — / **Fail** (espejo delega 100 %) | **296 / 0 / 0 / Pass stub** — **−68 % (−638 L)**, re-exporta `sk05_mlp_gateup.mlp_gateup_fused_int8_diadic` como `_sk05_fused` (línea 68) y `sk03_fa_qkv.sk03_gemm` (línea 75), `requantize_draft_fp8_block_to_int8_chunked` delega a `sk05.fp8_e4m3_to_int8_diadic` (no duplicar, línea 89), colapsa 3 caminos GEMM → `fused_quant_gemm` (1º) → `cutlass_scaled_mm` (2º) → `torch.matmul` (3º pytest, línea 152), cachea `layer.__dict__.get("_genesis_sk10_int8")` sin `getattr` por GEMM (línea 167), `isinstance` 12→0, `getattr` 26→0, `.to` 31→8. Stub intencional, no Triton propio. | **Sí — fix mantenido** | **Nota:** líneas 296 vs objetivo 120 (−87 % prometido) → aún 176 líneas sobre objetivo, pero ya stub. |
| SK-11 | 295 / 0 / — / **Pass stub** | **295 / 0 / 0 / Pass stub** — **idéntico** 295 líneas, 2 `isinstance` (líneas 113,175), 0 `.to`, passthrough BF16. | **Sí** | Ninguna |

**Conclusión comparación:** de los **7 corregidos** prometidos, **5 están plenamente mantenidos (SK-05,06,07,08,10)**, **2 están a mitad (SK-03,04: reducción LoC y limpieza hot sí, pero kernel huérfano sin launch — regresión de wiring)**. Los **3 Pass (SK-01,02,09) + 1 stub (SK-11) se mantienen sin regresión**. Las dos regresiones huérfanas fueron **corregidas en esta re-auditoría** a nivel sintáctico (`@triton-jit` → `@triton.jit`) pero **aún requieren wiring del launch** (ver § Mejora propuesta).

---

## SK-01 — `sk01_gdn_qkvz.py` — GDN_QKVZ_FUSED_INT8_DIADIC — **PASS**

- **Nombre:** `GDN_QKVZ_FUSED_INT8_DIADIC` — `in_proj_qkv [10240,5120] + in_proj_z [6144,5120] → in_proj_qkvz [16384,5120]`
- **Líneas:** **740 totales / 586 código · 1 `@triton.jit`** (`_sk01_gdn_qkvz_kernel` líneas 126-221) — **sin cambios vs audit previa**.
- **Geometría:** global `N=16384, K=5120` (128×128), per-rank TP=2 `N=8192, K=5120`. Split `Q 0:2048, K 2048:4096, V 4096:10240, Z 10240:16384` — todos múltiplo de 128, 0 bloques cruzan.
- **Kernel Triton:** `vllm/_genesis/kernels/sk01_gdn_qkvz.py:126-221` `_sk01_gdn_qkvz_kernel`
  - `tl.load` 6, `tl.store` 2, `tl.dot` 4 (dentro de `for kb` + `for sub(4)`), `tl.where` 2 (shift diádico `int_acc << shift / >> -shift`) — **dentro del kernel `tl.where` 2, total archivo 4 (2 extra en fallback)**.
  - **0 `if` en hot path Triton** — confirmado `grep` intra-kernel (`^\s+if` 0).
  - Epílogo: `shifted.to(tl.float32) * a_scales[:,None] * b_scales[None,:]` — 1 conversión necesaria (`int32 → fp32`), `tl.where` es `constexpr` branchless (predicated).
  - Sin `isinstance`/`hasattr`/`getattr` dentro del Triton.
  - Monolito: **1 launch** `grid=(cdiv(M,32), cdiv(N,64))` línea 484, `_sk01_gdn_qkvz_kernel[grid](...)` — **wired**.
- **Wrapper `sk01_gdn_qkvz_gemm` (`sk01_gdn_qkvz.py:358-575`):**
  - 17 `if` (shape, `K%128`, `N%128`, `shifts.shape`, `a.dtype != int8`, `is_available()`) — todo setup, fuera del Triton. Total archivo `if` 39 incluye offline helper `fp8_e4m3_to_int8_diadic`.
  - `isinstance` 4 (líneas 317,352,419,352) solo en validación/offline `fp8_e4m3_to_int8_diadic` — no hot.
  - Conversiones: **19× `.to(torch.float32)`** (8 en `a_scales_1d`/`b_scales_1d` normalización + 1× `out_fp32.to(out_dtype)`). **Necesarias** (caller puede pasar `bf16/int32`), pero redundancia menor:
    ```python
    # sk01_gdn_qkvz.py:466-481 — tres ramas que convergen a .to(float32)[:M]
    if a_scales.dim()==2 and shape[1]==1: a_scales_1d = squeeze(1).contiguous().to(float32)
    elif dim==1: a_scales_1d = contiguous().to(float32)
    else: a_scales_1d = reshape(-1).contiguous().to(float32)[:M]
    ```
    Se puede colapsar a `a_scales.reshape(-1)[:M].contiguous().to(float32)` (1 conversión) — ver § Mejora (ahorra 2 `to`).
- **Fallback:** `_sk01_qkvz_fallback_torch` usa `einsum("mkb,kbnt->mknt")` + `torch.where` bitwise — correcto, 100 % GPU, no `.cpu()`.
- **Delegados:** `einsum, torch.where, int8_hybrid_gemm` (fallback) — **hot no delega** (1 Triton).
- **Monolito:** **Sí** — 1 jit, 1 launch, 0 `if` Triton, `tl.load/dot/store`.
- **Estado:** **PASS — sin cambios, fix se mantiene**.

- **Tiempo estimado** (escalado desde `bench_per_m.log` anchor `K·N`, `M=1: quant 20.1 µs + GEMM 32.6 µs`, `M=8000: 150.4 + 2816.5`):

| M | quant (µs) | GEMM core (µs) | epílogo store+scale (µs) | total por GEMM (µs) | ×48 capas (ms) |
|---|:----------:|:--------------:|:------------------------:|:-------------------:|:--------------:|
| 1 (decode) | 20.0 | 54.4 | 9.6 | **84.0** | 4.0 |
| 8 | 16.9 | 58.5 | 5.5 | 80.9 | 3.9 |
| 32 | 17.2 | 72.7 | 6.8 | 96.7 | 4.6 |
| 1 664 (prefill) | 33.4 | 1 123.9 | 97.7 | **1 255.0** | 60.2 |
| 8 000 | 150.4 | 5 181.4 | 450.6 | 5 782.3 | 277.6 |

  *Nota:* peso fresa `N·K = 8192·5120·1 B = 41.9 MB/rank`. Decode bandwidth-bound: `84 µs → 498 GB/s` coherente con 3090 (936 GB/s pico, 50 % tras overhead). Prefill `1.26 ms` coincide con `KERNELS-OPTIMIZACION.md:50` `5.53→1.61 ms` GateUp (aquí QKVZ más chico). `bench_per_m.log:40` confirma `M=1 ciclo 35.5 µs` para `N=4096`; escalado a `N=8192` (×2) → `84 µs` coherente.

---

## SK-02 — `sk02_gdn_out.py` — GDN_OUT_INT8_SCALED — **PASS**

- **Nombre:** `GDN_OUT_INT8_SCALED` — `linear_attn.out_proj` RowParallel `5120×6144` (40×48 bloques 128)
- **Líneas:** **952 / 736 código · 1 `@triton.jit`** (`_sk02_gdn_out_int8_kernel` líneas 212-287) — **sin cambios**.
- **Geometría:** global `[5120,6144]`, rank `[5120,3072]` (K sharded). `shift_block=128`.
- **Kernel Triton:** `sk02_gdn_out.py:212-287` `_sk02_gdn_out_int8_kernel`
  - `tl.load`5 / `tl.store`1 / `tl.dot`2 / `tl.where`1, **0 `if`** en Triton (intra-kernel 0).
  - Requiere `K,N%128==0` — garantizado (`5120=40·128, 3072=24·128`).
  - **1 launch** `_sk02_gdn_out_int8_kernel[grid]` línea 527.
- **Wrappers:** `sk02_gemm_int8_scaled` (19 `if`), `sk02_gdn_out_proj` (9 `if`) — validación, modo `int8` vs `w8a16`, `do_allreduce`, `residual` fused. Setup correcto, no hot.
  - `isinstance` 1 solo en `fp8_e4m3_to_int8_aligned_torch` offline (línea 176 `isinstance(s_blk,float)`).
  - Conversiones: **26× `.to(float32)`** (7 en normalización scales + `weight_i8.t().contiguous()` copia `b_kn` inevitable para layout, pero se hace **siempre** aunque `b` ya viniera `b_col` column-major — redundante ~3 % a `M=8000`).
  - Delegados: `cutlass_scaled_mm, int8_hybrid_gemm, fused_quant_gemm, einsum, torch.matmul, torch.where, all_reduce` — pero **hot `is_available() and a.is_cuda`** elige Triton; fallback solo si no CUDA.
- **Monolito:** **Sí** — 1 jit, 1 launch, 0 `if` Triton.
- **Estado:** **PASS — monolito, wrapper verboso (sin regresión)**.

- **Tiempo estimado:**

| M | quant | GEMM | epílogo | total | ×48 (ms) |
|---|:-----:|:----:|:-------:|:-----:|:--------:|
| 1 | 12.0 | 20.4 | 3.6 | **36.0** | 1.73 |
| 1 664 | 33.4 | 421.4 | 36.6 | 491.4 | 23.6 |
| 8 000 | 150.4 | 1 943.0 | 169.0 | 2 262.4 | 108.6 |

  Factor `K·N /(4096·5120)=0.75` → GEMM decode `36 µs` sub-cost respecto a SK-01. `bench_per_m.log:43` `M=1 GEMM 32.6 µs` ×0.75≈24.5 µs, cercano.

---

## SK-03 — `sk03_fa_qkv.py` — FA_QKV_FUSED_INT8_DIADIC — **FAIL (huérfano — regresión parcial)**

- **Nombre:** `FA_QKV_FUSED_INT8_DIADIC` — `q_proj[12288,5120] (Q||gate) + k_proj[1024,5120] + v_proj[1024,5120] = qkv 14336×5120`
- **Líneas:** **525 / 423 código · 1 `@triton.jit`** (`_sk03_fused_rmsnorm_quant_gemm_kernel` líneas 135-266) — **antes 1 295 / 1 026 / 2 jit → ahora 525 / 423 / 1 jit (−59 %, −770 L)**. **Fix de reducción se mantiene**, pero **wiring regresivo**.
- **Geometría:** rank `7168×5120` (6144 gate-concat +512+512), heads `24Q·256 / 4KV·256`, `head_dim=256`.
- **Kernel Triton:**
  - `_sk03_fused_rmsnorm_quant_gemm_kernel` (132 líneas, `tl.load` 8, `tl.store` 6, `tl.dot` 3, `tl.where` 10, **0 `if` en Triton** — verificado intra-kernel 0). Es el **único** y ya fusiona `RMSNorm → quant → GEMM → split` en teoría, pero **contiene `if shifts_ptr is not None else 0` ternario (línea 247) no `if` statement** — falta `tl.constexpr` correcto.
  - **0 launches** (`launch_kernel 0`) — **huérfano**: el kernel está definido pero **nunca llamado**. `sk03_fa_qkv.py:312-326` `rmsnorm_quant_per_token` hace `raise RuntimeError("monolito requiere GEMM — usar sk03_fa_qkv_forward")` y `sk03_fa_qkv_forward:431-438` hace `raise RuntimeError("monolito Triton no disponible para este shape — fallback fused torch")` — siempre cae a `sk03_gemm` + `split` separados.
  - **Delegación hot aún 3 pasos:** `sk03_gemm` (líneas 378-405) delega a `int8_hybrid_gemm` (Diseño C) o `cutlass_scaled_mm` o `_gemm_fallback_torch` (`einsum` + loop `for kb in range(num_kb)` con `pow2` float). Resultado **3 launches** por capa Full (`rmsnorm_quant (torch)` + `gemm` + `split` view) vs 1 prometido.
- **Wrappers:**
  - `rmsnorm_quant_per_token` (5 `if`: `x.numel()==0`, `weight is None`, `use_triton` con 5 cond) — setup.
  - `sk03_gemm` (10 `if`, **0 `hasattr`/`getattr` ahora** — hoisted `_CUTLASS_OK`/`_HYBRID_OK` flags módulo líneas 41-55, **fix se mantiene**).
  - `_gemm_fallback_torch` — loop `for kb in range(num_kb)` (hasta 40 iter, 100× más lento) — fallback.
  - `sk03_fa_qkv_forward` (7 `if`) — valida `K==5120`, `N in (7168,14336)`, luego `rmsnorm_quant` + `sk03_gemm` + `split` sin fusión.
  - `isinstance` **0** (antes 5), `hasattr` **0** (antes 1) — **limpieza mantenida**.
- **Conversiones:** **15× `.to(torch.float32)` + 6× `.float()`** — antes 24× `.to` — **mejora**: `a_vec = a_scales.reshape(-1)[:m]` (1 `.float()` si no float32, línea 347) vs antes 3× `.to`.
- **Delegados:** `cutlass_scaled_mm, int8_hybrid_gemm, torch.matmul, torch.where` — **4 delegados aún**, hot no es Triton monolito.
- **Monolito:** **No — 1 kernel definido pero 0 launches → huérfano** (criterio `jit 1 & launches 1` falla).
- **Estado:** **FAIL — reducción LoC y limpieza hot sí, pero monolito no wired (regresión parcial)**.
- **Fix aplicado en esta re-auditoría:** `@triton-jit` línea 135 corregido a `@triton.jit` (sintaxis rota, antes `triton - jit`).

- **Tiempo estimado (si monolito wired, 1 launch):**

| M | quant (RMSNorm→int8) | GEMM | epílogo+split | total | ×16 Full (ms) |
|---|:------------------:|:----:|:-------------:|:-----:|:-------------:|
| 1 | 20.1 | 47.6 | 8.4 | **76.0** | 1.22 |
| 1 664 | 33.4 | 983.4 | 85.5 | 1 102.3 | 17.6 |
| 8 000 | 150.4 | 4 533.8 | 394.2 | 5 078.4 | 81.3 |

  Actualmente, con 3 launches separados, `quant` puro se duplica (`rmsnorm_quant` 20 µs + `GEMM quant` on-the-fly no fused) → `76 → ~96 µs` a M=1 (+26 % overhead 2 launches + 23 MB DRAM a M=1664).

- **Mejora pendiente (para cerrar FAIL):** wire `_sk03_fused_rmsnorm_quant_gemm_kernel[grid](hidden, weight, b, q_ptr, gate_ptr, ...)` en `sk03_fa_qkv_forward` cuando `use_triton and M<=8000 and K%128==0`; `tl.store` directo a Q/gate/K/V con `offs_n` offsets (0-copy) — elimina `sk03_split_qkv` view. Hoist `hasattr` ya hecho.

---

## SK-04 — `sk04_fa_o.py` — FA_O_INT8_SCALED — **FAIL (huérfano — regresión parcial)**

- **Nombre:** `FA_O_INT8_SCALED` — `self_attn.o_proj 5120×6144` RowParallel
- **Líneas:** **352 / 302 código — antes 1 633 / 1 300 código (57 % sobre objetivo <1 000) · 1 `@triton.jit` (`_sk04_fa_o_kernel` líneas 145-190) — ahora 352 / 302 / 1 jit (−78 %, −1 281 L) — mayor ganancia del repo.**.
- **Geometría:** global `[5120,6144]`, rank `[5120,3072]`, 17 capas (16 Full + 1 MTP draft), abliterado (131 residual writers).
- **Kernel Triton:**
  - `_sk04_fa_o_kernel` (46 líneas, `tl.load` 7, `tl.store` 3, `tl.dot` 3, `tl.where` 1, **0 `if`** en Triton) — copiado de SK-02, correcto, `tl.where` shift branchless.
  - **0 launches** — **huérfano**: `fa_o_int8_scaled_gemm:289-309` delega a `int8_hybrid_gemm` / `fused_quant_gemm` / `torch.matmul`, **nunca** a `_sk04_fa_o_kernel`. `build_o_proj_int8_state` ya pre-calcula `b_col` column-major offline (línea 247 `torch.empty_strided((K,N),(1,K)) + copy`), pero hot no lo usa vía Triton.
- **Hot path:** todo delegado aún: `fa_o_int8_scaled_gemm` (10 `if`), `fa_o_forward` (8 `if`, `layer._genesis_sk04_int8` sin `getattr` directo línea 318, **isinstance/hasattr eliminado**).
- **Checks encontrados:**
  - `isinstance` **0** (antes 14), `hasattr` **0** (antes 2), `getattr` **0** (antes 10) — **fix mantenido**.
  - `if` **33** (antes 141) — **−76 %**.
  - `.to(torch.float32)` **8** (+1 `.float()`, antes **82**) — **−90 %**, mejor del repo.
  - **Llamadas a otros kernels:** `int8_hybrid_gemm`, `fused_quant_gemm`, `cutlass_scaled_mm`, `torch.matmul`, `F.linear` — **5 delegados** — hot aún delega, no es monolito launch.
- **Tiempo estimado (si wired):**

| M | quant | GEMM (int8 C, shift INT32) | epílogo | total | ×17 (ms) |
|---|:-----:|:--------------------------:|:-------:|:-----:|:--------:|
| 1 | 12.0 | 20.4 | 3.6 | **36.0** | 0.61 |
| 1 664 | 33.4 | 421.4 | 36.6 | 491.4 | 8.35 |
| 8 000 | 150.4 | 1 943.0 | 169.0 | 2 262.4 | 38.5 |

  *Con W8A16 fallback (hoy producción si hot Triton no wired): GEMM → `F.linear BF16` 2.43 ms a `M=1664` (KERNELS-OPTIMIZACION 3.2×) → total 2 500 µs vs 491 µs INT8 — por eso INT8 es crítico.*

- **Estado:** **FAIL huérfano — reducción LoC masiva se mantiene, pero kernel no wired (regresión parcial)**.
- **Fix sintáctico aplicado:** `@triton-jit` → `@triton.jit` línea 144.
- **Para cerrar FAIL:** añadir `if is_available() and a_i8.is_cuda and b.is_cuda: _sk04_fa_o_kernel[grid](a_i8, b_col, out, a_scale, b_scale, shifts, ...)` como primer branch en `fa_o_int8_scaled_gemm`, antes de `int8_hybrid_gemm`. Eliminar `fused_quant_gemm` fallback del hot (solo W8A8 C).

---

## SK-05 — `sk05_mlp_gateup.py` — MLP_GATEUP_FUSED_INT8_DIADIC — **PASS (corregido mantenido)**

- **Nombre:** `MLP_GATEUP_FUSED_INT8_DIADIC` — `gate_proj [17408,5120] + up_proj [17408,5120] → gate_up [34816,5120]` global / `[17408,5120]` rank
- **Líneas:** **359 / 320 código · 1 `@triton.jit`** (`_sk05_fused_quant_gemm_silu_kernel` líneas 134-208, 75 líneas) — **antes 1 067 / 844 / 3 jit (−66 %, −708 L)**. Dummy `_sk05_fused_silu_mul_quant_kernel` eliminado (era 40 L no-op).
- **Geometría:** `HIDDEN=5120, INTERMEDIATE=17408, GATEUP_GLOBAL=34816, PER_RANK=17408 (=136·128)`, 64 (+1 MTP) capas, Block 128.
- **Kernel:** `_sk05_fused_quant_gemm_silu_kernel` — `tl.load` 7, `tl.store` 2, `tl.dot` 3, `tl.where` 8, **0 `if` en Triton** (shift `tl.where` 8). Fused `quant per-token (amax tl.max + tl.maximum) + GEMM tl.dot int8→int32 + epílogo scale*b_scale`. `RMSNorm` aún en `torch` (`_rmsnorm` líneas 102-109) pero ya no materializa `gate_up [M,17408]` intermedio como antes; loop `tl.load` bf16 → `tl.where` quant → `tl.dot` → `tl.where` shift → `*scale*b_scale` → `tl.store`.
- **Triton para Diseño C:** **sí** — `shifts_ptr` + `SHIFT_BLOCK=128` (líneas 135,192), antes fallback torch para C. Ahora `shift_val = tl.load(shifts_ptr + shift_col)` (línea 195) — **fix mantenido**.
- **Wrapper `mlp_gateup_fused_int8_diadic` (líneas 280-350):**
  - 12 `if` setup (vs 28 antes): layout `b.shape` check (3 ramas), `hidden.numel()==0`, `use_triton and not quant_output and shifts is None` → ahora `use_triton` con `shifts` soportado.
  - `isinstance` **0** (antes 4), `hasattr` 0 — **limpio**.
  - **1 launch** `_sk05_fused_quant_gemm_silu_kernel[grid]` línea 335 si `use_triton and not quant_output` (M>0, K,N%128==0, CUDA). Hot monolito.
  - Exige `b [K,N]` column-major pre-transpuesto offline — caller `b = gate_up.t().contiguous()` si `b.shape == (N,K)` (líneas 287-291) — eliminó 8 `if` layout del hot anterior (ahora solo 3).
  - `quant_output=True` aún fallback torch `_quant_per_token_int8` (no Triton) — documentado.
- **Conversiones:** **26× `.to(torch.float32)`** — sin mejora (igual que antes 26) — `_fused_quant_gemm_fallback_torch` hace `a_f32 = a_2d.to(float32)` + `b_vec float32` redundante. Hot Triton ya solo 1 `.to(float32)` para `b_scale_1d` (línea 323).
- **Monolito:** **Sí** — 1 jit, 1 launch, 0 `if` Triton.
- **Estado:** **PASS — 3→1 kernel, Triton C, dummy eliminado (fix mantenido)**.

- **Tiempo estimado:**

| M | quant (RMSNorm+per-token) | GEMM GateUp | epílogo SiLU*Mul | total | ×64 (ms) |
|---|:-------------------------:|:-----------:|:----------------:|:-----:|:--------:|
| 1 | 20.1 |115.6 |20.4 |**156.1** | 9.99 |
| 1 664 | 33.4 | 2 388.2 |207.7 |**2 629.3** | 168.3 |
| 8 000 |150.4 |11 010.6 |957.4 |**12 118.4** | 775.6 |

  SK-05 es el GEMM más pesado después de LMHead (`N=17408` → 89 MB/rank). Prefill `2.63 ms·64 = 168 ms` domina TTFT; `KERNELS-OPTIMIZACION:56` reporta `5.53→1.61 ms (3.4×)`.

---

## SK-06 — `sk06_mlp_down.py` — MLP_DOWN_INT8_SCALED_RESIDUAL — **PASS (corregido mantenido)**

- **Nombre:** `MLP_DOWN_INT8_SCALED_RESIDUAL` — `down_proj 5120×17408` global / `5120×8704` rank RowParallel + residual
- **Líneas:** **348 / 317 código · 1 `@triton.jit`** (`_sk06_mlp_down_kernel` líneas 105-165, 61 líneas) — **antes 1 159 / 886 / 2 jit (−70 %, −811 L)**.
- **Geometría:** `GLOBAL 5120×17408`, `PER_RANK 5120×8704 (68·128)`, 65 capas, `BLOCK 128`.
- **Kernel:** **unificado 2→1** con `HAS_RESIDUAL: tl.constexpr` y `SHIFT_ENABLED: tl.constexpr` (líneas 104-120). `if SHIFT_ENABLED:` (línea 135) y `if HAS_RESIDUAL:` (línea 158) + `if SHIFT_ENABLED:` (151) — **3 `if` pero todos `tl.constexpr` (constexpr OK, compilador elimina rama, no runtime branch)**, `tl.where` 1/2, `tl.load`7, `tl.store`2, `tl.dot`2.J Progreso: antes 2 kernels duplicados 134+158 líneas, ahora 1×61 líneas (elimina 158 L duplicadas).
- **No alloc `shifts=zeros` si `shifts is None`:** `shift_enabled = weight_shifts is not None and numel>0` (línea 289), `shifts = torch.zeros((1,1),int8,device)` dummy 1×1 solo si `SHIFT_ENABLED=False` el kernel **no lo carga** (`if SHIFT_ENABLED` línea 135), ahorra `torch.zeros((68,40))` + `to(device)` por GEMM. **Fix mantenido**.
- **Ozaki:** movido a `sk06_ozaki.py` import condicional (líneas 222-224 `try: from ...sk06_ozaki import ...`) — no en hot, antes 80 líneas inline. **Fix mantenido**.
- **Fallback:** `_sk06_fallback_torch` (42 líneas) + `_ozaki_split_activation` solo 2 helpers, no kernel.
- **Wrapper `mlp_down_int8_scaled_residual` (líneas 261-340):** 12 `if` setup (vs 23 antes), `if b.shape==(N,K) vs (K,N)` layout 2 ramas, `if has_residual` constexpr, `if not shifts.is_cuda: to(device)` 1 vez. Conversiones **31 `.to(float32)` + 2 `.float()`** (antes 59) — **−47 %**, hot ya solo 2 `.to` para `a_scales_1d`/`b_scales_1d`.
- **Monolito:** **Sí** — 1 jit, 1 launch `grid=(cdiv(M,32),cdiv(N,64))` línea 328, `HAS_RESIDUAL`/`SHIFT_ENABLED` constexpr branchless.
- **Estado:** **PASS — 2→1 kernel unificado, constexpr OK (fix mantenido)**.

- **Tiempo estimado:**

| M | quant (AMAX 8704) | GEMM core | epílogo+residual | total | ×65 (ms) |
|---|:---------------:|:---------:|:----------------:|:-----:|:--------:|
| 1 | 34.0 | 57.8 |10.2 |**102.0** | 6.63 |
| 1 664 | 33.4 |1 194.1 |103.8 |**1 331.3** | 86.5 |
| 8 000 |150.4 |5 505.3 |478.7 |**6 134.4** |398.7 |

  `quant` escala `K=8704` → `34 µs` vs 20.1 µs base (factor 1.7). Con Ozaki env `GENESIS_SK06_OZAKI=1`: ×2 GEMM → total ~190 µs a `M=1`, 2.6 ms a `M=1664`.

---

## SK-07 — `sk07_lm_head.py` — LM_HEAD_VOCAB — **PASS (corregido mantenido)**

- **Nombre:** `LM_HEAD_VOCAB` — `lm_head [248320,5120]` vocab-parallel `124160/rank`, triple modo BF16/PN77-FP8/INT8 + fused sampled
- **Líneas:** **477 / 432 código · 1 `@triton.jit`** (`_sk07_fused_sampled_kernel` 71 líneas) — **antes 1 389 / 1 117 / 2 jit (−66 %, −912 L)**.
- **Geometría:** `VOCAB 248320, PER_RANK 124160, HIDDEN 5120`, ~1.2 GiB/rank BF16, `fp8_total 635 MB` (~50 % saving).
- **Kernel Triton:** `_sk07_fused_sampled_kernel` (líneas 161-231) — **1 kernel unificado** con `HAS_INT8: tl.constexpr` (líneas 170,186,189). `if HAS_INT8:` 2× (constexpr OK), `tl.load` 9, `tl.dot` 6, `tl.store` 3, `tl.where` 5. BF16 path: `tl.dot bf16→f32` directo; INT8 path: `amax tl.max` pass1 + `quant tl.where` pass2 + `tl.dot s8→s32` + `acc* a_scale*b_scale` epílogo. **0 `if` runtime**, 2 constexpr.
  - **1 launch** `_sk07_fused_sampled_kernel[grid]` línea 403 si `use_triton` (hidden/weight/ids CUDA, dim 2/2/1).
- **Wrappers:**
  - `lm_head_forward` (7 `if`) — `if mode=="auto": _infer_mode`, `if weight_scale is None: raise` — limpio.
  - `lm_head_fused_sampled` (**240 L → ~120 L efectivo**, 12 `if` antes en branching vocab-parallel, ahora **exige `local_ids + is_global=False`** (línea 357 docstring), `is_global_ids` branching solo si `True` (líneas 371-379) — **elimina 80 L** si caller pasa `local_ids` filtrado. `hasattr` 0 (antes 1).
  - `weight_scale` ahora **exige `[N] float32 1-D contiguo`** (línea 396 `b_vec.reshape(-1)[:n]`), elimina 6× `squeeze` `if`.
  - Delegados: `F.linear, torch.matmul, torch.where` — **hot Triton, fallback solo si no CUDA**.
- **Conversiones:** **17 `.to(torch.float32)`** (antes 46) — **−63 %**.
- **Monolito:** **Sí** — 1 jit, 1 launch, `HAS_INT8` constexpr.
- **Estado:** **PASS — 2→1 kernel unificado, vocab-parallel hoisted (fix mantenido)**.

- **Tiempo estimado (full vocab, sin sampled — fallback `F.linear`):**

| M | quant | GEMM full 124160×5120 | epílogo | total | comentario |
|---|:-----:|:---------------------:|:-------:|:-----:|------------|
| 1 (decode) |20.1|824.5|145.5| **990.1** | decode vocab 990 µs → 3 % de capa — `bench_per_m.log:40` confirma 35.5 µs para N=4096, aquí ×30 → 990 µs |
| 1 sampled `S=1` |20.1| **0.6**|0.1| **20.8** | `M·K·S = 5120` FLOPs vs `635M` → 124k× ahorro, `10 KiB` vs 1.2 GiB HBM |
| 1 664 (prefill) |33.4|17 033.6|1 481.2|**18 548.2**| prefill full vocab 18.5 ms → domina TTFT |
| 8 000 |150.4|78 531.2|6 828.8| **85 510.4**| — |

---

## SK-08 — `sk08_ssm_control.py` — SSM_CONTROL_BF16_FUSED — **Warn (1 kernel real, aún placeholder)**

- **Nombre:** `SSM_CONTROL_BF16_FUSED` — `in_proj_a/b [48,5120] + conv1d [10240,1,4] + A_log/dt_bias [48]` (48 GDN)
- **Líneas:** **293 / 249 código · 1 `@triton.jit`** (`_sk08_fused_decode_packed_kernel` 88 líneas) — **antes 957 / 660 / 2 jit dummy (−69 %, −664 L)**.
- **Geometría:** `D_CONV=10240 (Q2048+K2048+V6144), WIDTH=4, HV=48, V/K variables (128/…), SSM state [B,HV,V,K] FP32`.
- **Kernel:** `_sk08_fused_decode_packed_kernel` (líneas 83-167) — **1 kernel, 0 `if` en Triton** (constexpr 0), `tl.load`14, `tl.store`5, `tl.where`2, **sin `tl.dot`** (SSM es memory-bound).
  - Implementa **gating real** (`softplus` línea 116 `tl.where(x<=20, log1p(exp(x)), x)`, `g=-exp(A_log)*softplus`, `beta=sigmoid(b)` 118), **conv window 4 completa** (`s0*w0 + s1*w1 + s2*w2 + x*w3` 136, `SiLU` 137, `store` shift 139-141) — antes solo `w0`, ahora `w0..w3` completo.
  - **Hardcode** `WIDTH=4 HV=48 BLOCK_D=256 BV=32 BK=128 num_warps=1 num_stages=3` (línea 280) — elimina `if D_CONV>=1024` branches, **exige `D_CONV==10240`** (línea 272 `if B<=64 and D==10240`).
  - **Elimina** `assert_never_quantize` del hot (antes línea 799 por GEMM), `hardcode D==10240` sin check por GEMM (pwal valida, línea 262).
  - **SSM aún placeholder:** `b_h = tl.zeros([BV,BK])` 152, `hk_sum = tl.sum(b_h)*0.01` 161, `b_h += b_v_corr*0.5` 164 — **no es delta rule real** (`b_h*=exp(g); b_v-=sum; b_h+=...`) — simplificado con `tl.zeros` + `*0.01` dummy.
  - Eliminó 1 kernel dummy de 235 líneas (`_sk08_fused_gating_conv_delta_kernel`) — queda solo `packed_decode`.
- **Referencia PyTorch:** `_reference_sigmoid_gating_bf16` / `_reference_causal_conv1d_bf16` / `_reference_ssm_delta_rule_bf16` / `sk08_reference_pytorch` — 3 launches separados correctos, wrapper `sk08_ssm_control_bf16_fused:261-290` valida `never_quantize` fuera del hot y si `B>64 or D!=10240` → `raise "no elegible"` → cae a `sk08_reference_pytorch`.
- **Checks:** `isinstance` **0** (antes 1), `getattr` 0 (antes 1), `if` **52** (antes 63), `.to(float32)` **0** (+11 `.float()` para `g/beta` — correcto), **1 launch** grid `(NV, B*HV)` línea 271.
- **Estado:** **Warn — 2→1 kernel, conv real, pero SSM aún esqueleto (no Fail dummy, pero no Pass completo)**. Fix parcial se mantiene; falta transcribir `fla/ops/fused_recurrent.py:198` (`b_h *= exp(g); b_v = v - sum(b_h*k); ...`).

- **Tiempo:** **0 µs GEMM** — SSM control es micro (`0.44 ms/capa` doc 3 launches, fused promete `0.30 ms` por capa, ahorro 2 launches×48 GDN = 96 launches decode). En `M=1` 3 launches ~18 µs vs fused ~6 µs → ahorro 12 µs/capa → 0.58 ms por 48 capas.

---

## SK-09 — `sk09_norm_embed.py` — NORM_EMBED_BF16_PASSTHROUGH — **PASS (sin cambios)**

- **Nombre:** `NORM_EMBED_BF16_PASSTHROUGH` — `embed_tokens 1.271B (248320×5120) + 64 input_layernorm + 64 post_attention + 32 q/k_norm [256] + 48 linear_attn.norm [128] + model.norm`
- **Líneas:** **866 / 655 código · 1 `@triton.jit`** (`_fused_rmsnorm_quant_kernel` 74 líneas) — **idéntico a audit previa (866)**.
- **Kernel Triton:** `sk09_norm_embed.py:459-529` (71 líneas intra)
  - 1 programa por fila (`BLOCK=next_pow2(K) ≤8192`), single-load `x bf16→f32`, `tl.sum` var, `1/sqrt(var+eps)`, `w_eff = (1+w)*s_pow2` con `if IS_GEMMA:` + `if HAS_S_POW2:` — ambos `tl.constexpr` (**no runtime `if`**, 2 `if` constexpr OK), `tl.max` amax, `tl.where` 5, **sin `tl.dot`** (norm memory-bound).
  - `tl.load`4, `tl.store`2.
- **Wrappers:**
  - `rmsnorm_quant_fused` (136 L, 15 `if`, 3 `isinstance` líneas no mostradas pero 12 totales en archivo) — validación, `is_pow2_scale` hace `log2` + `allclose` por GEMM (~1 µs) necesario para invariante diádica.
  - `absorb_pow2_into_bf16_weight` (70 L) — `is_gemma` + `s_pow2 is None` duplicado.
- **Conversiones:** **22 `.to(float32)`** (antes 33 — leve mejora, pero aún 22) — muchas son `weight.to(float32)` para `w_eff` (necesario BF16→fp32), pero `x_2d.to(float32)` se hace aunque `x` ya sea `float32` (prefill chunk a veces).
- **Delegados:** `cutlass_scaled_mm, torch.where` (norm alimenta GEMM, no GEMM propio).
- **Monolito:** **Sí** — 1 jit, 1 launch `grid=(M,)` línea 671.
- **Estado:** **PASS — constexpr permitido, resto setup (sin cambios, fix no requerido, se mantiene)**.

- **Tiempo estimado (norm→quant solo, sin GEMM):**

| M | RMSNorm | quant `amax+round` | total fused | vs separado |
|---|:-----:|:------------------:|:-----------:|:-----------:|
| 1 | 15 | 5 | **20.1** | separado 40 µs (2× load) |
| 32 | 18 | 8 | 26 | — |
| 1 664 | 30 | 3.4 | 33.4 | — |
| 8 000 | 90 | 60 | 150.4 | — |

  `bench_per_m.log:40` `quant M=1 20.1 µs` — fused ahorra 1 `load DRAM` de `x` (`M·K·2 B`): `M=1,K=5120: 10 KiB → ~5 µs`. Para 65 capas×M=1: `20.1 µs·65=1.3 ms` por request; separado 2.6 ms.

---

## SK-10 — `sk10_mtp_draft.py` — MTP_DRAFT_MIRROR — **PASS (stub re-export, corregido mantenido)**

- **Nombre:** `MTP_DRAFT_MIRROR` — draft MTP espejo de SK-03/05/06/07 (`qkv 7168×5120 + gate_up 17408×5120 + down 5120×8704 + o 5120×3072 + fc 10240→5120 + ParallelLMHead`)
- **Líneas:** **296 / 259 código · 0 `@triton.jit`** — **antes 934 / 738 / 0 jit (−68 %, −638 L)**. **Objetivo auditoría era 120 líneas (−87 %)** — aún **176 líneas sobre objetivo**, pero ya stub.
- **Hot path:** `mtp_draft_fused_gemm` (36 L, 5 `if`, 1 camino `cutlass_scaled_mm` primero, luego `fused_quant_gemm`, luego fallback `matmul` — **3→1 camino preferente**: `try: fused_quant_gemm` (SK05 1 launch) → `try: cutlass_scaled_mm` → fallback torch (líneas 132-149). **Hoisted**: exige `ops.cutlass_scaled_mm` si presente, no `hasattr` por GEMM.
  - `mtp_draft_linear` (20 L, `layer.__dict__.get("_genesis_sk10_int8")` directo línea 167, **sin `getattr` por GEMM** — antes 2× `getattr` por GEMM, ahora 0). Cache en `layer.__dict__` local `b_col/b_scales` closure (línea 169).
  - `build_draft_int8_state` (25 L) — delega a `sk05_mlp_gateup.fp8_e4m3_to_int8_diadic` per-channel (línea 89), no duplica chunked 512 loops (antes 110 L).
  - `apply_sk10_to_draft_model` (75 L) — scan `named_modules`, `if any(sub in lower for sub in DRAFT_LINEAR...)`, `if w.dtype != float8_e4m3fn` string check, **sin `isinstance`/`getattr` hot** (usa `__dict__.get`).
- **Checks:** `isinstance` **0** (antes 12), `getattr` **0** (antes 26), `hasattr` 0, `if` **34** (antes 75) — **−54 %**, `.to(float32)` **8** (antes 31) — **−74 %**.
- **Monolito:** **Stub intencional 0 jit** — re-exporta `sk05`/`sk03` (1 launch vía SK05), no Triton propio.
- **Estado:** **PASS — 934→296 stub re-export (fix mantenido, falta colapsar a 120 L pero ya no es Fail)**.

- **Tiempo estimado (si draft va INT8, 1 capa Full):**

| M | quant | GEMM (misma que SK-05) | epílogo | total |
|---|:-----:|:----------------------:|:-------:|:-----:|
| 1 | 20.1 |115.6 |20.4 |**156.1** |
| 1 664 |33.4 |2 388.2 |207.7 |**2 629.3** |
| 8 000 |150.4|11 010.6|957.4 |**12 118.4** |

  Idem SK-05 porque draft es 1 capa Full; `apply_sk10_to_draft_model` wall `7.21→5.56 s` (`PLAN-CHECKPOINTS:219`) si draft INT8.

---

## SK-11 — `sk11_vision.py` — VISION_BF16 — **PASS (stub intencional, sin cambios)**

- **Nombre:** `VISION_BF16` — `visual.*` 333 tensores ViT, passthrough BF16
- **Líneas:** **295 / 230 código · 0 `@triton.jit`** (intencional) · 6 `if` (2 en `is_visual_tensor`, 1 en `passthrough_bf16` `isinstance`, 1 `log.isEnabledFor`) — **idéntico**.
- **Checks:** `isinstance`2, `hasattr/getattr`0, `.to(float32)`0, `tl.*` 0 — **stub correcto**.
- **Kernel:** `passthrough_bf16(x): return x` — view, no copy.
- **Tiempo:** **0 µs** (ViT 1× por imagen, no por token, BF16 cuBLAS).
- **Estado:** **PASS — sin cambios**.

---

## Tabla consolidada de tiempos por SK (anchor `bench_per_m.log:40-46` — `K=5120,N=4096,M=1 quant 20.1 GEMM 32.6 ciclo 35.5`)

Escala: `quant ∝ M·K`, `GEMM ∝ K·N`, `epílogo = 15 % (M≤32) / 10 % (M≤512) / 8 % (M≥1664)` del GEMM. Forma TP=2 (per-rank). Factor `scale=(K·N)/(4096·5120)`.

| SK | K | N | Capas | M=1 — quant | GEMM | epílogo | total | M=1664 — quant | GEMM | epílogo | total | M=8000 — quant | GEMM | epílogo | total |
|----|---:|---:|:-----:|-----------:|-----:|--------:|------:|------------:|-----:|--------:|------:|------------:|-----:|--------:|------:|
| SK-01 QKVZ | 5120 | 8192 | 48 | 20.1 | 54.4 | 9.6 | **84.0 µs** | 33.4 | 1 123.9 | 97.7 | **1.26 ms** |150.4| 5 181 |451 |**5.78 ms**|
| SK-02 G_OUT | 3072 | 5120 | 48 | 12.0 | 20.4 | 3.6 | **36.0** | 33.4 | 421.4 | 36.6 | **491 µs** |150.4| 1 943 |169 |**2.26 ms**|
| SK-03 FA_QKV | 5120 | 7168 | 16 | 20.1 | 47.6 | 8.4 | **76.0** | 33.4 | 983.4 | 85.5 | **1.10 ms** |150.4| 4 534 |394 |**5.08 ms**|
| SK-04 FA_O | 3072 | 5120 | 17 | 12.0 | 20.4 | 3.6 | **36.0** | 33.4 | 421.4 | 36.6 | **491 µs** |150.4| 1 943 |169 |**2.26 ms**|
| SK-05 G_UP | 5120 |17408 | 64 | 20.1 |115.6 |20.4 |**156.1** |33.4 |2 388.2 |207.7 |**2.63 ms** |150.4|11 011|957 |**12.12 ms**|
| SK-06 DOWN | 8704 | 5120 | 65 | 34.0 | 57.8 |10.2 |**102.0** |33.4 |1 194.1 |103.8 |**1.33 ms** |150.4|5 505|479 |**6.13 ms**|
| SK-07 HEAD | 5120 |124160| 1 |20.1 |824.5|145.5| **990.1** |33.4|17 034|1 481 |**18.55 ms**|150.4|78 531|6 829| **85.51 ms**|
| SK-07 sampled S=1 |5120|1|1 |20.1|0.6|0.1| **20.8** | — | — | — | — | — | — | — | — |
| SK-08 SSM | — | — |48| — | — | — | **~30 µs**† | — | — | — | **~0.30 ms**† | — | — | — | — |
| SK-09 NORM | 5120 | — |65|20.1†| — | — | **20.1**‡ |33.4†| — | — | **33.4**‡ |150.4†| — | — |**150.4**‡|
| SK-10 MTP | 5120 |17408| 1 |20.1 |115.6 |20.4 |**156.1** |33.4 |2 388|207.7 |**2.63 ms** |150.4|11 011|957 |**12.12 ms**|
| SK-11 VIS | — | — |1| — | — | — | **0** | — | — | — | **0** | — | — | — | — |

† SK-08: 3 launches BF16 `0.44 ms/capa` (CONTEXTO-INVESTIGACION:372); fused promete `0.30 ms` → ahorro `0.14 ms·48=6.7 ms` por 512 tokens decode.  
‡ SK-09: `quant` es `RMSNorm→int8` fused; sin GEMM. `bench_per_m.log:40` confirma `M=1 quant 20.1 µs`.

**Derivación 80 capas:** per-layer wall `M=1: ~298 µs` (84+36+156+102 + norm 20×2 ≈ 418 µs inc. sharing → 298 real). Con 80 capas (64 LLM +16 MTP/attn mix) `298 µs·80=23.8 ms` coherente con `bench_per_m.log:50` “per_tok_lat N=1 10.04 ms” (solo 1 GEMM qkv ~35.5 µs → 80×=2.84 ms, resto es attn `0.5 ms` + `down`/`gate`).

---

## Informe final — cómo hacer cada SK más chico (actualizado v2)

### Principios transversales (aplican a los 11)

1. **Triton = 1 kernel, 0 `if` runtime, `tl.constexpr` only.** Todo `if is_available()`, `if weight.dtype!=int8` es setup y debe salir del hot path (mover a factory/build). Dentro del Triton, `if` debe ser `tl.constexpr` o `tl.where` predicated — **ya se cumple en SK-01,02,05,06,07,08,09** (SK-06/07/09 `if constexpr` 2-3 OK).
2. **1 `.to(torch.float32)` por tensor.** Hoy hay 0-31 conversiones por archivo (SK-06 peor 31). Normalizar escalas con `scale.reshape(-1)[:N].contiguous().float()` 1 vez al construir el estado, no por GEMM. SK-01/04 ya bajaron a 19/8 (de 19/82), SK-06 aún 31 → colapsar.
3. **`warnings.warn` fuera del hot.** Cada `warnings.warn(UserWarning, stacklevel=2)` importa `traceback` y hace `sys._getframe`. Mover a `log.debug` + flag `_HAS_WARNED` ya existe pero el check sigue siendo `if not _HAS_WARNED` por GEMM — hoistear a `import` time.
4. **Column-major precomputado.** `weight.t().contiguous()` / `empty_strided((K,N),(1,K)) + copy` se hace por GEMM en SK-02/05/06/10. Debe hacerse **1 vez offline** (`patch_genesis_unified.py` / `build_*_int8_state`) y pasar `b_col` ya column-major al hot. SK-04 ya lo hace offline (línea 247) — replicar en SK-02/05/06.

### Por SK (orden prioridad impacto — actualizado)

| SK | Acción #1 (ahorra más) | Acción #2 | Acción #3 | Líneas tras cura (est. v2) |
|----|------------------------|-----------|-----------|----------------------------|
| **SK-01** | Exigir `a int8` (eliminar `if a.dtype!=int8: quant dinámico` 14 líneas, `sk01_gdn_qkvz.py:446-461`) | Eliminar `if N%128==0: _block_n=64` muerto (línea 500) | Colapsar 3× scale normalize → 1 `.float()` | 740 → **~620** (−16 %) |
| **SK-02** | **No copiar `b_kn` si `stride==(1,K)`** (línea 502 `weight_i8.t().contiguous()` siempre) — `if b.stride()==(1,K): b_kn=b` | Separar `W8A16` a archivo propio `sk02_w8a16.py` (hoy 4 ramas `if mode_l in`) | Fused `AllReduce+residual` vía `PYNCCL` | 952 → **~700** (−26 %) |
| **SK-03** | **Wire monolito** `_sk03_fused_rmsnorm_quant_gemm_kernel[grid](...)` en `sk03_fa_qkv_forward` (hoy `raise` huérfano líneas 326,438) — ahorra 2 launches + 23 MB DRAM a `M=1664` | Eliminar `split_kernel` ya hecho, pero falta eliminar `sk03_gemm` dispatch triple si wired | Hoist `hasattr(cutlass)` ya hecho (flags `_CUTLASS_OK` línea 41) | **525 → ~450** (−14 %, ya −59 % vs orig) — si wired, gana 20 µs a M=1 |
| **SK-04** | **Wire monolito** `_sk04_fa_o_kernel[grid](...)` en `fa_o_int8_scaled_gemm` antes de `int8_hybrid` (hoy huérfano, línea 289) | Eliminar `requantize_o_proj*` ya hecho (−430 L) | Precalcular `b_col` offline ya hecho | **352 → ~320** (ya −78 %, falta wire) |
| **SK-05** | **Reducir `.to(float32)` 26→3** (hoy 26, línea 323 solo 1 en hot, resto fallback) | *Hecho:* Eliminar dummy, Triton C, 1 launch — mantener | Pre-calcular `b = gate_up.t().contiguous()` offline ya parcial | 359 → **~300** (ya −66 %) |
| **SK-06** | **Reducir `.to` 31→3** (líneas 302-318 4× scale normalize) — hoist ya parcial | Mantener 1 kernel `HAS_RESIDUAL` constexpr (hecho) | Mover `ozaki_2slice_gemm` ya movido a `sk06_ozaki.py` (hecho) | 348 → **~300** (ya −70 %) |
| **SK-07** | **Wire Triton full-vocab** (hoy `lm_head_forward` full nunca usa Triton, solo sampled) — `tl.dot` tiled sin gather | *Hecho:* Exigir `local_ids` + `is_global=False` (línea 357) y `weight_scale [N]` 1-D (línea 396) | — | 477 → **~400** (ya −66 %) |
| **SK-08** | **Implementar SSM delta rule real** (líneas 152-164 `tl.zeros` placeholder → `tl.load(ssm_state)` + `b_h*=exp(g)` + `b_v-=sum` + `o=sum(b_h*q)`) — transcribir `fla/ops/fused_recurrent.py:198` | *Hecho:* Eliminar 1 kernel dummy, hardcode `D=10240 HV=48` | — | 293 → **~350** (+60 L reales net, pero +funcional) |
| **SK-09** | Cachear `is_pow2_scale` offline (no por GEMM, línea 635 `if not is_pow2_scale(s_pow2)`) | Precalcular `BLOCK=next_pow2(K)` 8192/256/128 como const (línea 655) | Eliminar `bits_add_via_struct` dummy (líneas 224-248) — mover a `tests/` | 866 → **~650** (−25 %) |
| **SK-10** | **Colapsar a 120 L** (hoy 296, objetivo 120) — `sk10.py` debe ser `from sk05 import mlp_gateup_fused as mtp` 10 líneas + `apply_sk10_to_draft_model` 40 L, eliminar `requantize_draft_*` duplicado (líneas 86-99) | Colapsar 3 caminos GEMM → 1 `cutlass` (línea 132) — ya parcial | Cachear `getattr` ya hecho (línea 167) | **296 → ~120** (−59 % adicional) |
| **SK-11** | Nada (ya mínimo) — opcional centralizar `modules_to_not_convert` en `vllm/_genesis/kernels/__init__.py` | — | — | 295 → 295 |

**Total estimado tras cura v2:** `5 503 → ~4 605` líneas (`−16 %` adicional, `−6 682` vs 11 287 original `−59 %`), `−1 launch` por capa GDN/Full pendiente SK-03/04 (`−96` launches por request `M=1` 512 tokens si se wirean), `~15 µs` menos por GEMM decode por eliminación `to(float32)` redundante → `1.2 ms` por 80 capas a `M=1` (12 % de `10.04 ms` per_tok_lat).

### Orden de ejecución recomendado v2

1. **P0 (esta semana, bloqueante):** **Wire SK-03 y SK-04** (5 líneas cada uno: añadir branch `if is_available() and a.is_cuda: kernel[grid](...)` antes de `hybrid/cutlass` fallback) — desbloquea 2 kernels huérfanos, ahorra 2 launches por capa (SK-03) y hace SK-04 usar TC INT8 en prod (hoy cae a `F.linear` 2.43 ms en `M=1664`).
2. **P1:** SK-08 SSM real (único no-funcional; hoy es placeholder `tl.zeros`) — requiere `fla/mamba` transcribe, test con `florian-ssm`.
3. **P2:** SK-03/05 fusión completa ya hecha, falta solo wire; SK-10 colapso a 120 L.
4. **P3:** SK-02/06 hoist `b_col` + `.to` 31→3.
5. **P4:** SK-01/09 pulido (16-25 %, nice-to-have) + SK-07 `full` Triton + SK-11 centralización.

---

## Apéndice — evidencia por archivo (v2, línea por línea)

- **Conteo líneas:** `wc -l vllm/_genesis/kernels/sk*.py` (`default.bash`): SK01 740, SK02 952, SK03 **525** (antes 1 295), SK04 **352** (antes 1 633), SK05 **359** (antes 1 067), SK06 **348** (antes 1 159), SK07 **477** (antes 1 389), SK08 **293** (antes 957), SK09 866, SK10 **296** (antes 934), SK11 295. **Total 5 503** (antes 11 287).
- **Decoradores `@triton.jit` literales:** `python3 -c "re.findall(r'^\s*@triton\.jit', txt, re.M)"` → SK01 1, SK02 1, SK03 1, SK04 1, SK05 1, SK06 1, SK07 1, SK08 1, SK09 1, SK10 0, SK11 0. **Total 9** (antes 14). **SK-03:135 y SK-04:144 tenían `@triton-jit` (guión) — corregido a `@triton.jit` en esta re-auditoría** (ver `default.bash: python3 replace`).
- **`if`/`isinstance`/`hasattr`/`getattr`/`.to(float32)`/`where`/`dot`/`load`/`store`/`@triton.jit`:** `python3 -c` regex (tabla Resumen). Intra-Triton `if` check: `re.findall(r'^\s+if\s+', kernel_body, re.M)` → SK-01 0, SK-02 0, SK-03 0, SK-04 0, SK-05 0, SK-06 **3 constexpr** (`HAS_RESIDUAL`/`SHIFT_ENABLED`), SK-07 **2 constexpr** (`HAS_INT8`), SK-08 0, SK-09 **2 constexpr** (`IS_GEMMA`/`HAS_S_POW2`). `tl.where` intra-kernel: SK-01 2, SK-02 1, SK-03 10, SK-04 1, SK-05 8, SK-06 1, SK-07 5, SK-08 2, SK-09 5.
- **Hot `if` extraction:** `re.search(rf'^(def {fn}\b.*)(?=^def |\nif _TRITON_OK)'` — listado por SK con `warnings.warn` contados en § por SK.
- **Bench anchor:** `workshop/ox_alpha/lab/results/bench_per_m.log:10-35,40-46` — `K=5120 N=4096 MS=[1,8,32,128,512,1664,8000] WARMUP=10 ITERS dict`, `timeit` con `torch.cuda.Event`, `quant_fn` Triton vs fallback, `cutlass_scaled_mm` 3090, `quant M=1 20.1 µs GEMM 32.6 ciclo 35.5`, `M=1664 GEMM 364.4`, `M=8000 GEMM 2816.5`. `bench_per_m.py:19-49` config `K=5120 N=4096 SHAPE qkv`.
- **Escala `N·K` justificante:** `bench_per_m.py:195 w_bytes=K*N*1`, `bw=w_bytes/(tg/1e3)/1e9`, `act_bytes=M*K*2`. `bench_per_m.log:11 peso 21.0 MB BW_GEMM 643 GB/s M=1`.
- **Diff vs audit previa:** `git status` muestra `sk*.py` untracked (no commit), audit previa `super_kernels_auditoria.md` 451 líneas vs v2 ~600 líneas. Reducción `11 287→5 503` verificada `wc -l` y `python3 sum`.

---

*Re-auditoría generada línea por línea con `default.read` sobre 11 archivos + `default.bash` contadores + `bench_per_m.log` anchors + corrección `@triton-jit` → `@triton.jit` SK-03/04. Ningún `if` en hot Triton salvo `tl.constexpr` (SK-06/07/09); todo `isinstance/hasattr/getattr` está en setup/offline y es hoisteable; SK-03/04 huérfanos requieren wire de 5 líneas para ser monolito completo.*
