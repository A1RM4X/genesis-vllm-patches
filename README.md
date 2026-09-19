<p align="center">
  <img src="assets/logo.png" alt="Genesis vLLM Patches" width="780">
</p>

# Genesis vLLM Patches

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![vLLM](https://img.shields.io/badge/vLLM-0.29.0-orange.svg)](https://github.com/vllm-project/vllm)
[![Patches](https://img.shields.io/badge/patches-146-green.svg)](docs/PATCHES.md)
[![GPU](https://img.shields.io/badge/GPU-2%C3%97%20RTX%203090%20(sm__86)-purple.svg)](docs/HARDWARE.md)

**Runtime patches for [vLLM](https://github.com/vllm-project/vllm) that make a 27B hybrid-GDN
model serve at 256K context on two consumer RTX 3090s — integer attention kernels, a custom
Marlin W4A8 path, speculative decoding, and a tiered KV cache.**

> This is a fork of [Sandermage/genesis-vllm-patches](https://github.com/Sandermage/genesis-vllm-patches),
> re-targeted from Qwen3.6 / RTX A5000 to **Qwen3.8-27B on 2× RTX 3090**, and moved from vLLM
> 0.27.1 to **0.29.0**. Upstream's patch framework, dispatcher and much of the patch set are
> theirs; the PTX kernels and the patches numbered PN120+ are this fork's.

---

## What this is

A **drop-in patcher**, not a fork of vLLM. It pins to a known vLLM version and applies small,
surgical changes — text edits at named anchors, class-rebind wrappers, and in-process
registrations — that turn a stock vLLM into a production server for a hybrid
Gated-DeltaNet model on hardware vLLM upstream does not target.

Every patch is independently gated, states in its own docstring *why* it exists and what it
measured, and **retires itself** when upstream merges the underlying fix: patches carry
`upstream_drift_markers` and skip cleanly once they detect their own obsolescence.

What it is **not**: a fork of vLLM, a quantizer, an inference engine, or a training framework.

---

## Production configuration

The container serving production today:

```
compose/docker-compose.qwen38-27b-noon-dflash2-v029.yml    →  container: genesis-27b-dflash2
```

| | |
|---|---|
| Model | `noon-at-cgn/Qwen3.8-27B-Uncensored-W4A16-AutoRound` — hybrid GDN, 48 linear-attention + 16 full-attention layers |
| Engine | vLLM 0.29.0 + Genesis |
| Hardware | 2× RTX 3090 (sm_86), TP=2, PCIe, no NVLink |
| Speculative decoding | **DFlash2** W4A16 drafter, `num_speculative_tokens=8` |
| Weights / activations | fp16 · W4A8 Marlin (`VLLM_MARLIN_INPUT_DTYPE=int8`) |
| KV cache | `int8_per_token_head`, served by this project's integer decode kernel |
| Context | 262 144 tokens · `max-num-seqs 10` · `gpu-memory-utilization 0.92` |
| **KV capacity** | **566 314 tokens** — 2.16× concurrency at full context |

A second compose keeps the **MTP** drafter as the fallback path:
`compose/docker-compose.qwen38-27b-noon-w4a8-v029.yml` → container `genesis-27b-v029`.

### Measured performance

Measured with the `bench.sh` harness from
[club-3090](https://github.com/noonghunna/club-3090): 3 warm-ups + 5 measured runs,
cache-busted prompts. Raw output and the config that produced it live in
[`tests/bench/resultados/`](tests/bench/resultados/).

| | mean | CV |
|---|---|---|
| decode, narrative | **131.2** tok/s | 4.5% |
| decode, code | **257.6** tok/s | 2.4% |
| prefill @ 10K | **2801** tok/s | 2.8% |
| prefill @ 90K | **1828** tok/s | 5.4% |

Behavioural quality is checked with the same project's `quality-test.sh --quick`
(ToolCall-15 + InstructFollow-15): **27/30**, with `pass@k=3` reporting zero flaky failures.

---

## Getting started

### 1. Credentials

Every compose reads its API key from the environment. **Nothing secret is versioned** —
`.env` is gitignored and only `.env.example` is tracked.

```bash
cp compose/.env.example compose/.env
openssl rand -hex 32          # generate a real key, paste it into compose/.env
```

Benchmark and diagnostic scripts read the same `VLLM_API_KEY` **with no default**: a script
run without it fails loudly instead of sending the wrong key.

### 2. Run

```bash
cd compose
docker compose -f docker-compose.qwen38-27b-noon-dflash2-v029.yml up -d
docker inspect genesis-27b-dflash2 --format '{{.State.Health.Status}}'
```

The healthcheck **generates a token** rather than pinging `/health`. A server that boots but
produces garbage is reported unhealthy — that has caught real failures here.

### 3. Verify

```bash
# every patch decision, with its reason
docker logs genesis-27b-dflash2 2>&1 | grep "Genesis Dispatcher"

# KV capacity actually obtained
docker logs genesis-27b-dflash2 2>&1 | grep "GPU KV cache size"
```

---

## How the patch system works

`vllm/_genesis/dispatcher.py` holds **146 patch entries**. Each declares an env flag, a
default, a category, and a credit note recording the problem it solves and what was measured.
At boot, `apply_all.py` walks them and prints one line per decision — `APPLY`, `SKIP` or
`DRIFT` — with the reason.

Patches reach vLLM three ways:

| mechanism | when it applies |
|---|---|
| **Text patch** at a named anchor | the change lives inside a vLLM function body |
| **Class rebind** | a method can be wrapped or subclassed instead |
| **In-process registration** | a backend or hook must exist in the *serving* process |

The third exists because `apply_all` runs as a separate process and then `exec`s
`vllm serve`: anything registered in memory is lost crossing that boundary. `PN143` installs a
single hook inside `load_general_plugins()` — which runs in every process — and everything
in-process hangs off it.

**Anchor patching fails silently when upstream moves a line:** the patch reports `SKIPPED`,
the server boots without it, and the only symptom is lower throughput. Before migrating to a
new vLLM version run:

```bash
python3 -m vllm._genesis.preflight_anclajes --bajar v0.29.0
```

It checks every anchor against the target tree without downloading an image, and reports which
patches would break *and actually run in this configuration*.

---

## Repository layout

```
vllm/_genesis/        the patches, kernels and dispatcher — the project itself
  ├─ wiring/          one module per patch, grouped by subsystem
  ├─ kernels/         PTX and Triton kernels (integer attention, Marlin, GDN)
  └─ tests/           unit tests for the patch machinery
compose/              the two production composes + .env.example
tests/bench/
  ├─ medicion/        one script per question (acceptance, KV blocks, concurrency)
  └─ resultados/      raw benchmark output, with the config that produced it
docs/                 patch ledger, hardware notes, architecture
benchmarks/           historical measurement campaigns
```

---

## Status and open problems

Kept deliberately honest — these are the things a reader would otherwise discover the hard way:

- **vLLM 0.29.0 is validated here but is not on the inherited pin allowlist.** Boot logs a
  pin-gate warning; that is expected, not a fault.
- **About 8.5% of boots on 0.29.0 came up broken** — the model emits two tokens and stops.
  The generation-based healthcheck catches it. Root cause still open.
- **KV offload reads now work, but only on a 100% prefix hit.** Until 2026-09-19 the tiers
  wrote 50 GB per run and the lookup had never returned a single hit. Three of this
  project's own patches were responsible, and all three are fixed: `PN97` refused every
  L3→L2 promotion once L2 was permanently full (disk tier queried 32 times vs 7 347 for
  RAM); `PN91`'s 0.2 s deferral budget was shorter than a disk promotion, so an in-flight
  block was scored as a miss and vetoed the whole lookup; and `PN81`'s 30 GB quota made L3
  **smaller than L1** while pruning by write age. Rescue of an evicted 20K prompt now costs
  **0.97 s instead of 6.76 s (−85.6%)**, returning 612 MiB over `CPU_to_GPU`.
  What is still open: `_lookup_complete_chunks` requires *all nine* groups to hit, and
  upstream stores only the reachable tail of a sliding-window group. So when the attention
  prefix hits partially (16 of 20 chunks at production sizing), the DFlash2 drafter's group
  misses at that truncated boundary and vetoes the other eight groups' hits. Full analysis
  and the two candidate fixes are in `vllm/_genesis/diag_offload.py`.
- **PCIe links negotiate gen4 ×8, not ×16** on this machine. Every number above was measured
  under that constraint.

---

## Credits

Built on [Sandermage/genesis-vllm-patches](https://github.com/Sandermage/genesis-vllm-patches).
Several fixes and the whole measurement methodology come from
[noonghunna/club-3090](https://github.com/noonghunna/club-3090), whose benchmark and quality
harnesses are used here directly.

Apache 2.0 — see [LICENSE](LICENSE).
