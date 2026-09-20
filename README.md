<p align="center">
  <img src="assets/logo.png" alt="Genesis vLLM Patches" width="780">
</p>

# Genesis vLLM Patches

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![vLLM](https://img.shields.io/badge/vLLM-0.29.0-orange.svg)](https://github.com/vllm-project/vllm)
[![status](https://img.shields.io/badge/status-in%20development-yellow.svg)](#status-and-open-problems)
[![GPU](https://img.shields.io/badge/GPU-2%C3%97%20RTX%203090%20(sm__86)-purple.svg)](docs/HARDWARE.md)

**A personal, work-in-progress fork of runtime patches for
[vLLM](https://github.com/vllm-project/vllm), built around one idea: push
`int8` through *every* stage of the pipeline on hardware that has no `fp8`
tensor cores.**

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

### Where int8 turns out to be the floor, not a waypoint

The same measurement discipline says where to stop. These are all negative
results from this rig, and they are the reason the table above says int8 and not
int4:

- **KV in int4** gives 2× the cache but costs **−27% prefill and −21% long
  decode**, and degrades long generations. int8 per token-head has 4× less error
  than fp8 at the same width.
- **All-reduce in int4** is **7× worse than int8** at equal traffic. The path
  forward there is overlapping the transfer, not compressing it harder.
- **W4A4 in the MLP** composes its errors: `gate_up` alone ≈ fp16, but both
  projections together lose 0.1 logprob.
- **The decode GEMMs are already at the DRAM roof** — 98–103% of the achievable
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

Behavioural quality is checked with the same project's `quality-test.sh --quick`
(ToolCall-15 + InstructFollow-15): **27/30**, with `pass@k=3` reporting zero
flaky failures.

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

The most involved subsystem, and the most honest about its limits. Full write-up
in **[docs/KV-OFFLOADING.md](docs/KV-OFFLOADING.md)**.

Three of this project's own patches had it deadlocked: it wrote 50 GB per run
and had never returned a single hit. `PN97` refused every L3→L2 promotion once
L2 was permanently full; `PN91`'s 0.2 s deferral budget expired before the
asynchronous promotions resolved, and in strict mode an in-flight block scores
as a miss, vetoing the whole lookup; `PN81`'s quota made L3 **smaller than L1**
while pruning by write age.

With all three fixed, rescuing an evicted 20K prompt costs **1.21 s instead of
6.88 s (−82.5%)**, reproducible across three runs from a clean disk.

**But** — under live traffic it still returns **0 external hits**, because
`_lookup_complete_chunks` requires all nine KV-cache groups to hit and real
traffic produces partial matches that get discarded wholesale. L1 carries the
load at 76.4%. Do not enable this expecting a win yet.

---

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

- **KV offload returns 0 external hits under real traffic.** The mechanism works
  and the synthetic rescue is reproducible, but the all-groups-must-hit
  constraint means partial matches are wasted. See
  [docs/KV-OFFLOADING.md](docs/KV-OFFLOADING.md) §4.
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
