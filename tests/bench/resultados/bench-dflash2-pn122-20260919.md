# bench.sh — Qwen3.8-27B W4A16 + DFlash2, 2× RTX 3090 (TP=2), 2026-09-19

Salida cruda del `scripts/bench.sh` de club-3090, sin editar:
[`bench-dflash2-pn122-20260919.txt`](bench-dflash2-pn122-20260919.txt)

## Qué se midió

| | |
|---|---|
| Modelo | `noon-at-cgn/Qwen3.8-27B-Uncensored-W4A16-AutoRound` (híbrido GDN: 48 linear_attention + 16 full_attention) |
| Engine | vLLM **0.29.0** + parches Genesis |
| Borrador | **DFlash2** W4A16 (`incoai/Qwen3.8-27B-DFlash2`, 5 capas), `num_speculative_tokens=8` |
| Hardware | 2× RTX 3090 sm_86, PCIe (sin NVLink), driver 610.57.04, cap 229 / 243 W |
| dtype | fp16, `--kv-cache-dtype int8_per_token_head`, `VLLM_MARLIN_INPUT_DTYPE=int8` |
| Contexto | `--max-model-len 262144`, `--max-num-seqs 10`, `gpu-memory-utilization 0.92` |
| KV | **566.314 tokens**, concurrencia máxima 2,16× a 262.144 |

## Resumen

| | media | CV |
|---|---|---|
| decode narrativa | **131,2 tok/s** | 4,5% |
| decode código | **257,6 tok/s** | 2,4% |
| prefill 10k | **2801 tok/s** | 2,8% |
| prefill 90k | **1828 tok/s** | 5,4% |
| PP del engine @90k | 8862 tok/s | 1,9% |

## Parches Genesis que afectan estos números

* **PN145** — el grupo `SlidingWindowSpec` del borrador elegía `block_size=16` para una página
  padeada a 0,87 MB, y se llevaba 1153 de 2001 bloques por request. Corregido:
  178.823 → 405.080 tokens de KV.
* **PN146** — `group_size` de los grupos de KV elegible; con 8 en vez del 5 que da la
  heurística de upstream (que toma el mínimo, o sea las 5 capas del borrador).
* **PN122** — rollback del GDN con cinta de rango 1 en vez de K copias del estado:
  `num_speculative_blocks = 0`. +7,6% de KV por −2,6% de decode (A/B intercalado).
* **PN131** — decode de atención entero (SK-18) sobre KV int8 por token-cabeza.
* **PN130** — Marlin propio con escalas int16 con signo (el de upstream rompe con las escalas
  negativas de AutoRound).

## Dos cosas del entorno que el script marcó y conviene mirar

1. **PCIe a x8, no x16.** `GPU0/GPU1 gen=4/4 width=8/16` en los dos, con el aviso de
   slot/riser/BIOS bifurcation. El pico de rx en decode es 5193 MB/s = 38,8% del denominador
   de x8. En este rig eso importa: el all-reduce de TP ya estaba cerca del techo del enlace.
2. **Desbalance entre placas.** Al cerrar la corrida, GPU0 al 0% y GPU1 al 99%; durante el
   decode `sm_mean = 97,6%` en las dos. Coincide con que el paso lo fija la placa más lenta.

La sección `CAPTURE: DRAFT ACCEPTANCE` dice "no drafter output" porque scrapea un formato de
log que vLLM 0.29.0 no emite; las métricas de spec-decode sí están al final del archivo
(accept-len 2,0–2,25 en esa ventana, que es prosa con sampling —temperature 0,6— y no el
código: con código y greedy da 5,3–5,7).
