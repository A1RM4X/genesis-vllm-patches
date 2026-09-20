<p align="center">
  <img src="assets/logo.png" alt="Genesis vLLM Patches" width="780">
</p>

# Genesis vLLM Patches

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![vLLM](https://img.shields.io/badge/vLLM-0.29.0-orange.svg)](https://github.com/vllm-project/vllm)
[![status](https://img.shields.io/badge/status-in%20development-yellow.svg)](#status-and-open-problems)
[![GPU](https://img.shields.io/badge/GPU-2%C3%97%20RTX%203090%20(sm__86)-purple.svg)](docs/HARDWARE.md)

**A personal, work-in-progress fork of runtime patches for
[vLLM](https://github.com/vllm-project/vllm), built around one idea: make
`int8` the *compute* standard through every stage of the pipeline — on hardware
that has no `fp8` tensor cores. Next target: `int4` in the KV cache.**

## Read this first

This is **not a product**. It is a working notebook with code attached:

- **It targets one machine.** 2× RTX 3090 (sm_86), TP=2, PCIe gen4 ×8, no
  NVLink, 30 GB of host RAM, one specific quantized checkpoint. Numbers,
  thresholds and several patches are tuned to exactly that. On different
  hardware, expect some patches to be useless and others to be wrong.
- **It is in active development.** Patches land, get measured, and sometimes get
  deleted when the measurement says the idea was wrong. The git history contains
  retractions on purpose — see [Dead ends](#dead-ends).
- **It pins to vLLM 0.29.0.** Patching is by text anchors, so a different vLLM
  version will silently skip patches rather than fail loudly. There is a
  preflight tool for that (below), and it should be run before any bump.
- **Documentation is bilingual.** Code comments and commit messages are in
  Spanish; the docs under `docs/` are being moved to English.

> Forked from [Sandermage/genesis-vllm-patches](https://github.com/Sandermage/genesis-vllm-patches),
> re-targeted from Qwen3.6 / RTX A5000 to **Qwen3.8-27B on 2× RTX 3090**, and
> moved from vLLM 0.27.1 to **0.29.0**. Upstream's patch framework, dispatcher
> and most patches below PN120 are theirs; the PTX kernels and everything from
> **PN120 up** are this fork's.

---

## The thesis: int8 end to end

Ampere (sm_86) has **no `fp8` tensor cores**. Everything modern inference
assumes — fp8 KV caches, fp8 GEMMs, fp8 collectives — either falls back to
emulation or simply is not there. What the 3090 *does* have is a full-rate
`int8` tensor path (`mma.m16n8k32.s8`) and an `int4` one (`mma.m16n8k64.s4`,
measured at 2.0× the int8 TOPS).

So the bet of this fork is to stop treating int8 as a storage format and use it
as the **compute** format, stage by stage:

| stage | stock vLLM here | this fork | how | measured |
|---|---|---|---|---|
| Weights | int4 | int4 | AutoRound / GPTQ checkpoint | — |
| GEMM activations | fp16 | **int8** | `VLLM_MARLIN_INPUT_DTYPE=int8` + **PN130** (own Marlin, signed int16 scales) | **+57%** prefill |
| KV cache | fp16 / fp8 | **int8 per token-head** | `--kv-cache-dtype int8_per_token_head` | **+30%** decode @50K vs fp8, and **4× less error** |
| Attention decode | fp16/fp8 kernel | **integer, hand-written PTX** | **PN131** — SK-18h reads the int8 KV directly, no dequant | parity with FlashInfer fp8 on quality, decode and PP |
| TP all-reduce | fp16 | **int8** | **PN120** — half the bytes over a PCIe link already at 96% | **+6.5%** prefill |
| Drafter KV | inherits fp16 | **int8** | compose | part of the **+217%** KV capacity |

The point is the *composition*. Any one of these is a known trick; doing all of
them means a tensor never has to leave the integer domain between the RMSNorm
that produces it and the attention that consumes it — which is also why the
KV cache can be int8 without a dequant step in front of the attention kernel.

### What each KV format actually costs, measured

The least intuitive claims here are about the KV cache, so: numbers first. This
is the error of the attention **output** against float — what reaches the next
layer, not the error of the stored tensor. Measured on real dumps from this
model (`PN123`, layers 3 and 35, 8 204 tokens of code, 192 queries seeing
almost the whole context). Reproduce with
[`tests/proto/kv_escalas_grupo_eval.py`](tests/proto/kv_escalas_grupo_eval.py).

| KV scheme | bits/value | layer 3 | layer 35 |
|---|---|---|---|
| `fp8_e4m3`, per-tensor scale | 8.00 | 1.27% | 2.30% |
| `int8` per token-head | 8.10 | 0.31% | 0.79% |
| **`int8` per token-head + Hadamard** ← in production | 8.10 | **0.22%** | **0.39%** |
| `int4` per token-head | 4.10 | 6.49% | 12.61% |
| `int4` per token-head + Hadamard | 4.10 | 4.08% | 6.68% |
| `int4` group 64 + Hadamard | 4.25 | 3.54% | 5.46% |
| `int4` group 32 + Hadamard | 4.50 | 3.15% | 4.96% |
| `int4` group 16 + Hadamard | 5.00 | 2.71% | 4.33% |

Three things fall out of that table.

**int8 beats fp8 by ~4× at the same width.** The reason is where the bits go.
`e4m3` spends 4 of its 8 bits on an exponent, buying dynamic range this tensor
does not use: post-RoPE vectors with qk-norm are nearly isotropic and each
(token, head) slice occupies a narrow range. int8 with a scale fitted *per
token-head* spends all 8 bits on mantissa inside exactly that range. The 0.1 bit
is that scale, amortized over the head.

**Hadamard rotation helps int8 — roughly halving the error at layer 35** (0.79%
→ 0.39%). This corrected an earlier claim in this README that said rotation did
nothing for int8; that claim came from misreading a study which had compared
Hadamard against WUSH, *both* rotated, never against no rotation at all.
`PN126` is now **on** in production.

**Per-group scales rescue a lot of int4, but not enough.** Going from
per-token-head to group-32 with rotation takes layer 35 from 12.61% to 4.96%.
That is a 2.5× improvement for 0.4 extra bits — real, and the direction the
quantization work should take. It is still an order of magnitude worse than
int8, which is why int4 is not in production.

### What that costs in throughput

`bench.sh`, same harness and depths as above, one full run per arm:

| | PN126 off | PN126 on | |
|---|---|---|---|
| prefill @ 10K | 2829 ±5 | 2791 ±38 | **−1.4%** |
| prefill @ 90K | 1850 ±11 | 1859 ±5 | +0.5% |
| decode, narrative | 131 ±6 | 135 ±6 | +3.2% |
| decode, code | 235 ±22 | 258 ±12 | +9.5% |

Read that honestly: only the 10K prefill difference is outside the noise (that
arm's CV was 0.2%). The decode numbers move in the right direction but sit
inside a 4–9% run-to-run spread, and this is **one run per arm** — not enough to
claim a decode gain. The defensible statement is that rotation costs at most
~1.4% of prefill and buys a measurable accuracy improvement.

And on Ampere, int8 KV is also simply **faster than fp8**, which was not the
goal — the switch was made looking for quality:

| | fp8 | int8 |
|---|---|---|
| decode @1K | 114.7 | **157.8** tok/s |
| decode @50K | 91.8 | **119.7** tok/s (+30%) |
| prefill @50K | 2251 | 2206 tok/s |

Two reasons: Triton **does not accept fp8 on SM86** at all, so that path pays
conversions the hardware cannot do natively; and the integer kernel consumes the
int8 KV directly, with no dequant in front of it. The honest cost is ~7% of KV
capacity, because the integer kernel reserves its own accumulators.

### Next: int4 in the KV cache

int8 is the standard the pipeline runs on today. **int4 in the KV cache is the
target**, and the reason it is not in production yet is worth stating precisely,
because it is not what you would guess.

The kernel is not the problem. Ampere has a native `int4` tensor path
(`mma.m16n8k64.s4`, measured at **2.0×** the int8 TOPS, same register layout as
`s8`), the integer attention kernel has an int4 variant, and a full working
compose exists in this repo's history (`9305c90`). **Quality is the problem.**

The numeric side is in the table above: the best int4 scheme measured so far
(group-32 + Hadamard, 4.5 bits) sits at 3.15% / 4.96%, against 0.22% / 0.39%
for the int8 that runs in production. An order of magnitude. It also costs
**−27% prefill and −21% long decode** against int8.

But the blocking symptom is not numeric at all — it is behavioural: **the model
runs on and never closes.**
On a long coding task (a Tetris with SRS, 7-bag, hold and T-spin), three runs
each: fp8 produced 6 111 / 6 594 / 7 780 tokens and passed 6/6 execution checks;
int4 produced 11 153 / 32 000 / 32 000 — the last two hitting the `max_tokens`
ceiling — and passed 4/6, 2/6, or emitted no code at all.

**Where it stands.** Per-group scales along the head dimension were the
outstanding lead, and they have now been measured (the table above): they take
layer 35 from 12.61% to 4.96%, a 2.5× improvement for 0.4 extra bits. Real, and
the right direction — but not enough on its own. What is still untried: group
smoothing folded into the group scales and the RMSNorm (2–4.8× more precise than
dynamic, at no cost), per-layer decisions about whether rotating helps, and
learned rather than fixed rotations. Prototypes under `tests/proto/sk18_a0*`.

So the open work is **an advanced quantization process aimed at int4 quality**,
not another kernel. And its oracle cannot be per-layer numeric error or a short
answer — it has to be a long generation task where you check whether the model
*terminates*, because that is the failure mode.

### Where int8 really is the floor

Not everything below 8 bits is worth chasing. These are negative results from
this rig, and they are settled:

- **All-reduce in int4** is **7× worse than int8** at equal traffic. The path
  there is overlapping the transfer, not compressing it harder.
- **W4A4 in the MLP** composes its errors: `gate_up` alone ≈ fp16, but both
  projections together lose 0.1 logprob.
- **Dictionary / VQ methods on the KV** (PQ, RVQ, Lexico-style sparse coding)
  all lose to plain int4 at the same width. Post-RoPE vectors with qk-norm are
  nearly isotropic — there is no structure left for a codebook to exploit.
- **The decode GEMMs are already at the DRAM roof** — 98–103% of achievable
  bandwidth. No amount of PTX buys anything there; the remaining 2× is in
  unpacking nibbles, not in arithmetic.

---

## What is running right now

```
compose/docker-compose.qwen38-27b-noon-dflash2-v029.yml   →  genesis-27b-dflash2
```

This is the container serving today, and the one to copy if you want a
known-good starting point.

| | |
|---|---|
| Model | `noon-at-cgn/Qwen3.8-27B-Uncensored-W4A16-AutoRound` — hybrid GDN, 48 linear-attention + 16 full-attention layers |
| Engine | vLLM 0.29.0 + Genesis |
| Hardware | 2× RTX 3090 (sm_86), TP=2, PCIe, no NVLink |
| Speculative decoding | **DFlash2** W4A16 drafter, `num_speculative_tokens=8` |
| Weights / activations | int4 weights · **int8 activations** (W4A8 Marlin) |
| KV cache | **`int8_per_token_head`**, read directly by this project's integer PTX decode kernel |
| Context | 262 144 tokens · `max-num-seqs 10` · `gpu-memory-utilization 0.92` |
| **KV capacity** | **566 314 tokens** — 2.16× concurrency at full context |
| Address on this host | `172.20.0.228:8320`, alias `vllm-server` |

A second compose keeps the **MTP** drafter as a fallback path:
`compose/docker-compose.qwen38-27b-noon-w4a8-v029.yml` → `genesis-27b-v029`.
The two are mutually exclusive: they share the IP and the `vllm-server` alias,
so whichever is up owns the endpoint.

### Measured performance

Measured 2026-09-20 with the `bench.sh` harness from
[club-3090](https://github.com/noonghunna/club-3090), run **directly against the
container** (no reverse proxy in the path). Decode: 3 warm-ups + 5 measured
runs. Prefill: 1 warm-up + 3 measured runs, cache-busted with a fresh haystack
per run. Raw output in [`tests/bench/resultados/`](tests/bench/resultados/).

| | mean | CV |
|---|---|---|
| prefill @ 10K | **2829** tok/s | 0.2% |
| prefill @ 90K | **1850** tok/s | 0.6% |
| decode, narrative | **131** tok/s | 4.8% |
| decode, code | **235** tok/s | 9.4% |

Prefill is very stable. Decode spread is real, not noise: throughput tracks
DFlash2's acceptance rate, which depends heavily on the content being generated
— code accepts far more draft tokens than prose, which is the whole 131 → 235
gap.

The harness reports `INTEGRITY: OK` and `swap check: PASS`. Its draft-acceptance
and engine-timing captures came back empty (the engine does not log acceptance
at this verbosity), so the acceptance rate behind that gap is inferred from
throughput, not measured here.

### Behavioural quality

`quality-test.sh --quick` from the same project (ToolCall-15 +
InstructFollow-15), sampled at `temperature=0.6` — so expect run-to-run spread:

| run | stack | score | failing scenarios |
|---|---|---|---|
| 2026-09-19 | before today's changes | **27/30** | IF-04, TC-05, TC-09 |
| 2026-09-19 | same | **27/30** | IF-04, IF-10, TC-05 |
| 2026-09-20 | int8 + PN126 | **25/30** | IF-04, IF-10, TC-05, TC-09, **TC-11** |

Read the scenarios, not the total. **IF-04 and TC-05 fail in all three runs** —
those are stable failures, not noise. IF-10 and TC-09 each fail in one of the
two older runs, so they are the flaky ones, and today they simply happened to
fail together. Only **TC-11** is new.

That is one run against the current stack, and the 27/30 baseline predates
today's changes, so the two-point difference is **not** attributable to any
single patch yet. A clean A/B — PN126 on and off, several runs each, nothing
else moved — is pending.

### KV format: capacity and throughput

Same harness, same depths, one full run per format, everything else at the
production configuration:

| | `auto` (fp16) | `int8_per_token_head` |
|---|---|---|
| **KV capacity** | 328 160 tok · 1.25× | **566 314 tok · 2.16×** |
| prefill @ 10K | 2681 (CV 5.1%) | 2648 (CV 13.3%) |
| prefill @ 90K | 1690 (CV 1.3%) | **1838** (CV 0.7%) |
| decode, narrative | 151 (CV **23.3%**) | 134 (CV 3.3%) |
| decode, code | 250 (CV **16.2%**) | 247 (CV 9.4%) |

**int8 gives +73% more KV capacity than fp16** — 1.25× concurrency becomes
2.16× at full context. On throughput it wins clearly at 90K prefill (+8.8%);
everywhere else the means are close but the *stability* is not. fp16 runs the
generic kernel and swings 16–23% between runs; the integer kernel stays at
3–9%. Compare means with the CV beside them, not alone.

### Where a decode step actually goes

Kernel-by-kernel profile of one decode step, so the table above has a shape
behind it. Two HTML reports live in `docs/`:
[radiography](docs/informe-radiografia-decode-2026-09-16.html) (every kernel:
where it lives, µs/step, %, what it does) and
[remaining headroom](docs/informe-margen-por-seccion-2026-09-16.html) (eight
fronts ranked by prize, with the literature for each).

| section | share of the step |
|---|---|
| Marlin linears | **43%** (of which `lm_head` alone: 12.6%) |
| Attention — own PTX kernel | 19.2% |
| TP all-reduce | 11.2% |
| GDN (linear-attention layers) | 9% |
| Draft model, running fp16 | 7.6% |
| everything else | 10% |

Prefill has a different shape entirely: NCCL 40%, Marlin 38%, attention 8.6% —
which is why PN120 (int8 all-reduce) pays off in prefill and barely registers in
decode.

⚠️ **This profile is outdated in two ways that matter.** It was taken
2026-09-16 at 57K context with **int4 KV and the MTP drafter at K=4**;
production now runs **int8 KV and DFlash2 at K=8**. Expect the attention and
drafter shares in particular to have moved. It is kept because the *ranking* has
held up — Marlin linears dominate, `lm_head` is a surprisingly large single
item, and the drafter running in fp16 is free money left on the table (PN133
would fix it and is written but not enabled). Re-profiling against the current
configuration is on the list.

---

## What this fork adds

The dispatcher holds **103 patch entries**; **56 are active** in the production
container. Everything from PN120 up is this fork's work. ✅ marks what is
actually running in production, taken from the dispatcher's boot log rather than
from the code defaults.

### The integer pipeline

| | | |
|---|---|---|
| ✅ | **PN130** | Own Marlin W4A8 with **signed int16 scales** (vllm#48905). Upstream's Marlin produced silent garbage on AutoRound checkpoints with negative scales |
| ✅ | **PN125** | Positive-scale fallback for Marlin W4A8-INT8, kept as a safety net |
| ✅ | **PN131** | **SK-18h**: attention decode in hand-written PTX, integer throughout, reading `int8_per_token_head` KV with no dequant |
| ✅ | **PN120** | TP all-reduce compressed to int8 during prefill |
| ✅ | **PN124** | Fast TRITON_ATTN on Ampere for head dim 256 |
| ✅ | **PN126** | Hadamard rotation of q/k after RoPE — halves the int8 KV error at layer 35, costs ~1.4% prefill |
| | PN134-137 | int8 activation quantization fused into the RMSNorm; group smoothing folded into the weights (exact SmoothQuant) |
| | PN133 | Draft-model linears in W8A8 — the 7.6% above |
| | PN139 | `lm_head` in per-group int4 — the 12.6% above |

### KV cache and capacity

| | | |
|---|---|---|
| ✅ | **PN145** | Sliding-window block aligned with primary attention |
| ✅ | **PN146** | Selectable KV group size — upstream's heuristic picks badly on hybrid models |
| ✅ | **PN122** | MTP rollback on GDN without speculative blocks |
| ✅ | **PN127** | `MambaManager` honours `drop_eagle_block` (vllm#48375) |
| ✅ | **PN121** | Preemption-cascade guard with deferred frees |

PN145 + PN146, plus giving the drafter int8 KV, took capacity from 178 823 to
**566 314 tokens (+217%)**. The mechanism is counter-intuitive and worth
knowing: on a hybrid model `bytes_per_block` is the **max** across groups, not
the sum, so *smaller* groups yield *more* total blocks.

### Speculative decoding

| | | |
|---|---|---|
| ✅ | **PN142** | DFlash2 usable on 0.29.0 (vllm#51581 + port of `fa5017a5`) |
| ✅ | **PN144** | Scales the DFlash2 drafter's residual so it fits fp16. The drafter ships in bf16 and overflowed at 50 080 against fp16's 65 504 ceiling, accepting **0%** while producing perfectly coherent text. Watch the `eps` trap: RMSNorm is scale-invariant, `eps` is not |
| ✅ | **PN128** | Async input-prep waits for spec-decode post-processing (GDN+MTP) |
| | PN132 | Trimmed vocabulary for the drafter (FR-Spec) |

### Infrastructure

| | | |
|---|---|---|
| ✅ | **PN143** | Genesis hooks into *every* vLLM process via `load_general_plugins()` |
| ✅ | **PN83** | The engine explains its own memory layout and risks at boot |
| ✅ | **PN81/PN88** | Disk quota and Prometheus metrics for the KV tiers — vLLM ships neither |

PN143 exists because `apply_all` runs as a separate process and then `exec`s
`vllm serve`: anything registered in memory is lost across that boundary. This
bit us silently once — the server booted, logged "applied", and ran the generic
kernel anyway. Only a one-time notice inside the kernel's `forward` revealed it.

### Tiered KV cache (RAM + NVMe)

When VRAM fills, vLLM **discards** old prefix blocks and recomputes them later.
This subsystem sinks them to RAM (L2) and NVMe (L3) instead, and brings them
back over PCIe. Full write-up in
**[docs/KV-OFFLOADING.md](docs/KV-OFFLOADING.md)**.

Measured, rescuing a 20K prompt that had been evicted from the GPU:

| | |
|---|---|
| recompute from scratch | 6.88 s |
| **rescued from L2/L3** | **1.21 s (−82.5%)** |
| returned over `CPU_to_GPU` | 612 MiB |
| reproducibility | 3 runs, clean disk and restart each time |

Getting there required fixing three of this project's own patches, which had
the read path deadlocked: `PN97` refused every L3→L2 promotion once L2 was
full; `PN91`'s 0.2 s deferral budget expired before the asynchronous promotions
resolved, and in strict mode an in-flight block scores as a miss, vetoing the
whole lookup; and `PN81`'s quota had made L3 smaller than L1.

#### Size L2 relative to L1 — this is the whole game

**L2 is not a cache in front of L3. It is the gateway to it.** Secondary tiers
cannot touch GPU memory: every L3→L2→GPU promotion has to land in L2 first. So
L2's size sets three things at once — how much can leave L1 without falling
straight through to disk, how much of a prefix can be reassembled at once, and
therefore whether a lookup can hit at all.

The lookup needs a **complete** chunk-aligned prefix across every KV group. A
partial prefix is not a partial win, it is no win. And you cannot assemble a
prefix larger than L2.

So the sizing rule is in **tokens**, not bytes, and it is anchored to L1:

```
cpu_bytes_to_use  ≥  (your GPU KV cache size in tokens)  ×  17.5 KiB
```

Take the `GPU KV cache size` your engine prints at boot and read across:

| your L1 (GPU KV) | L2 at **1×** — minimum | at 1.5× — comfortable | at 2× — room for concurrency |
|---|---|---|---|
| 100 000 tok | 1.7 GiB | 2.5 GiB | 3.3 GiB |
| 250 000 tok | 4.2 GiB | 6.3 GiB | 8.3 GiB |
| **566 314 tok** ← this rig | **9.5 GiB** | 14.2 GiB | 18.9 GiB |
| 1 000 000 tok | 16.7 GiB | 25.0 GiB | 33.4 GiB |
| 2 000 000 tok | 33.4 GiB | 50.1 GiB | 66.8 GiB |

Below 1× the tier still works, it just cannot hold a whole prefix, so hits
become partial and partial hits are discarded. This rig runs **2.0 GiB = 0.21×**,
because the host has 30 GB total and the container already sits at 16.4 GB.

The rule is not theoretical — hit rate tracks L2 size directly. Chunks of the
attention prefix that hit, out of 20:

```
L2 = 2 GiB (0.21× L1)   →   0, 8, 12 or 16 of 20, varying run to run
L2 = 6 GiB (0.64× L1)   →   16 of 20
complete prefix         →   the 1.21 s rescue above
```

On this machine L1 is 566 314 tokens, so L2 wants 9.4 GiB. The host has 30 GB
total with the container already at 16.4 GB, so that is not reachable here —
which is why this rig runs at 0.21× and why the tier is provisioned to exist
rather than to pay. **On a machine with RAM to spare, size L2 at or above L1 and
this becomes a straight win.** On this one it is correctly configured for the
memory available, and the ceiling is RAM, not the mechanism.

#### When it engages at all

It only pays when the working set **exceeds** L1 — that is the entire premise.
Under the traffic observed on this rig it does not:

```
num_preemptions_total     0          ← nothing was ever evicted from L1
kv_cache_usage_perc       0.0
prefix tokens seen        395 636    ← against an L1 of 566 314
```

With no eviction there is nothing to bring back, and the stores you see
(2.78 GiB written) are the *proactive* copies the design makes while blocks are
being computed — working exactly as intended. Before concluding anything about
hit rate, check `num_preemptions_total` first: if it is zero, the tier was never
asked to do its job.

## Getting started

### 1. Credentials

Every compose reads its API key from the environment. **Nothing secret is
versioned** — `.env` is gitignored, only `.env.example` is tracked.

```bash
cp compose/.env.example compose/.env
openssl rand -hex 32          # generate a real key, paste it into compose/.env
```

Benchmark and diagnostic scripts read the same `VLLM_API_KEY` **with no
default**: a script run without it fails loudly instead of sending a stale key.

### 2. Run

```bash
cd compose
docker compose -f docker-compose.qwen38-27b-noon-dflash2-v029.yml up -d
docker inspect genesis-27b-dflash2 --format '{{.State.Health.Status}}'
```

The healthcheck **generates a token** rather than pinging `/health`. A server
that boots but produces garbage is reported unhealthy — that has caught real
failures here.

### 3. Verify

```bash
# every patch decision, with its reason
docker logs genesis-27b-dflash2 2>&1 | grep "Genesis Dispatcher"

# KV capacity actually obtained
docker logs genesis-27b-dflash2 2>&1 | grep "GPU KV cache size"

# the integer attention backend actually took over
docker logs genesis-27b-dflash2 2>&1 | grep "PN131"
```

That third one matters: a patch can report `applied` and still not run. See
PN143 above.

### Before bumping vLLM

Anchor patching **fails silently when upstream moves a line**: the patch reports
`SKIPPED`, the server boots without it, and the only symptom is lower
throughput. Run:

```bash
python3 -m vllm._genesis.preflight_anclajes --bajar v0.29.0
```

It checks every anchor against the target tree without downloading an image, and
reports which patches would break *and actually run in this configuration*.

---

## Status and open problems

Kept deliberately honest — these are the things you would otherwise discover the
hard way:

- **int4 KV is the open goal, and quality is the only blocker.** The kernel,
  the hardware path and a working compose all exist; what is missing is a
  quantization process built around per-group scales along the head dimension.
  See [Next: int4 in the KV cache](#next-int4-in-the-kv-cache).
- **L2 is at 0.21× L1 and wants 9.4 GiB to do its job.** The tiered KV cache
  works and the rescue is reproducible; what limits it here is host RAM, not the
  mechanism. See [Size L2 relative to L1](#size-l2-relative-to-l1--this-is-the-whole-game).
- **The decode profile is a configuration behind.** Re-profile against int8 KV +
  DFlash2 K=8 before acting on the percentages above.
- **The drafter still runs in fp16** — 7.6% of every decode step. PN133 is
  written and not enabled.
- **About 8.5% of boots on 0.29.0 came up broken** — the model emits two tokens
  and stops. The generation-based healthcheck catches it. Root cause still open.
- **vLLM 0.29.0 is validated here but is not on the inherited pin allowlist.**
  Boot logs a pin-gate warning; that is expected, not a fault.
- **PCIe links negotiate gen4 ×8, not ×16** on this machine. Every number above
  was measured under that constraint.
- **The 3090s are power-capped at 220 W**, which drops the sustained SM clock to
  810 MHz and the real bandwidth roof to 640 GB/s, not 730. Any burst
  measurement on this rig overestimates by ~11%.

### Dead ends

Recorded so they are not repeated. Both came from single runs and both had to be
retracted:

- **"The draft group vetoes the other groups' hit, remove the veto."** A patch
  (PN147) was written, applied, and measured with a successful rescue — n=1, and
  it did not reproduce. With PN91's budget raised and PN147 *off*, the same group
  hits 19/19. The data was always there. PN147 was deleted.
- **"PN100's staging ring wastes 44% of L2, disabling it gains capacity."**
  Worse: the attention prefix went from 12/20 to 0/20 hits. Bounding ephemeral
  traffic is what protects the long thread's prefix.

This subsystem has genuine run-to-run variance. Take every result with a clean
disk, a restart, and at least two repetitions.

---

## Repository layout

```
vllm/_genesis/        the patches, kernels and dispatcher — the project itself
  ├─ wiring/          one module per patch, grouped by subsystem
  ├─ kernels/         PTX and Triton kernels (integer attention, Marlin, GDN)
  └─ tests/           unit tests for the patch machinery
compose/              the two engine composes + .env.example
tests/
  ├─ bench/medicion/  one script per question (acceptance, KV blocks, concurrency)
  ├─ bench/resultados/raw benchmark output, with the config that produced it
  └─ repro/           standalone harnesses that reproduce upstream bugs without a GPU
docs/                 subsystem write-ups, hardware notes, decode profiles
benchmarks/           historical measurement campaigns
```

Each patch states in its own docstring *why* it exists and what was measured.
Patches carry `upstream_drift_markers` and retire themselves once they detect
that upstream merged the underlying fix.

---

## Credits

Built on [Sandermage/genesis-vllm-patches](https://github.com/Sandermage/genesis-vllm-patches).
Several fixes and the whole measurement methodology come from
[noonghunna/club-3090](https://github.com/noonghunna/club-3090), whose benchmark
and quality harnesses are used here directly.

Apache 2.0 — see [LICENSE](LICENSE).
