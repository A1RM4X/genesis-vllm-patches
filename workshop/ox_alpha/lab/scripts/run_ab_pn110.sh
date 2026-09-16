#!/usr/bin/env bash
# CK-2.1/2.2 — A/B de PN110 (requant INT8 + despacho por fase).
# Workload PREFILL-dominado: 2 prompts de ~8000 tokens, 64 tokens de salida.
# Métricas: GEN_DONE wall/tps, aceptancia SpecDecoding (debe moverse poco:
# el rejection sampler verifica) y STEP times en lab_<pid>.log por pata.
set -uo pipefail
mkdir -p /lab/results/ab_pn110/baseline /lab/results/ab_pn110/pn110

run_one() {
  local TAG=$1; shift
  echo ""
  echo "########## A/B: $TAG ##########"
  env "$@" LAB_RESULTS_DIR=/lab/results/ab_pn110/$TAG \
    LAB_LOG_STATS=1 LAB_LOG_ALL_STEPS=1 \
    LAB_PROMPTS=2 LAB_TOKENS=64 LAB_PROMPT_LEN=8000 \
    python3 /lab/scripts/trace_real.py 2>&1 \
    | tee "/lab/results/ab_pn110/${TAG}.log" \
    | grep -iE "gen_done|prompt_len|SpecDecoding metrics|PN110" || true
}

run_one baseline LAB_GPU_MEMORY_UTILIZATION=0.80 env -u LAB_ENABLE_PN110
run_one pn110   LAB_GPU_MEMORY_UTILIZATION=0.70 LAB_ENABLE_PN110=1 GENESIS_ENABLE_PN110_INT8_PHASE_DISPATCH=1

echo ""; echo "### A/B PN110 DONE ###"
exit 0
