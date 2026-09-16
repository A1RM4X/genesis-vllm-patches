#!/usr/bin/env bash
# A/B aislado SOLO PN110 vs Marlin puro (sin B2/B3/B5/B7)
set -uo pipefail
mkdir -p /lab/results/ab_pn110_solo/baseline /lab/results/ab_pn110_solo/pn110

run_one() {
  local TAG=$1; shift
  echo ""
  echo "########## A/B SOLO: $TAG ##########"
  env "$@" LAB_RESULTS_DIR=/lab/results/ab_pn110_solo/$TAG \
    LAB_LOG_STATS=1 LAB_LOG_ALL_STEPS=1 \
    LAB_PROMPTS=2 LAB_TOKENS=64 LAB_PROMPT_LEN=8000 \
    python3 /lab/scripts/trace_real.py 2>&1 \
    | tee "/lab/results/ab_pn110_solo/${TAG}.log" \
    | grep -iE "gen_done|prompt_len|SpecDecoding metrics|PN110" || true
}

run_one baseline LAB_GPU_MEMORY_UTILIZATION=0.80 env -u LAB_ENABLE_PN110
run_one pn110   LAB_GPU_MEMORY_UTILIZATION=0.80 LAB_ENABLE_PN110=1 GENESIS_ENABLE_PN110_INT8_PHASE_DISPATCH=1

echo ""; echo "### A/B PN110 SOLO DONE ###"
