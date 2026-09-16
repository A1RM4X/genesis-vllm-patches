#!/usr/bin/env bash
# CK-1.2a — A/B de PN109 (buffers persistentes de metadatos spec-decode).
# Métricas: GEN_DONE tps + aceptancia (deben ser IDÉNTICOS: el parche es
# numéricamente transparente) y tiempos STEP por execute_model en
# <LAB_RESULTS_DIR>/lab_<pid>.log de cada pata (análisis post-corrida).
set -uo pipefail
mkdir -p /lab/results/ab_pn109/baseline /lab/results/ab_pn109/pn109

run_one() {
  local TAG=$1; shift
  echo ""
  echo "########## A/B: $TAG ##########"
  env "$@" LAB_RESULTS_DIR=/lab/results/ab_pn109/$TAG \
    LAB_LOG_STATS=1 LAB_LOG_ALL_STEPS=1 \
    LAB_PROMPTS=${LAB_PROMPTS:-10} LAB_TOKENS=${LAB_TOKENS:-250} \
    python3 /lab/scripts/trace_real.py 2>&1 \
    | tee "/lab/results/ab_pn109/${TAG}.log" \
    | grep -iE "gen_done|SpecDecoding metrics|PN109" || true
}

run_one baseline env -u LAB_ENABLE_PN109
run_one pn109   LAB_ENABLE_PN109=1

echo ""; echo "### A/B PN109 DONE ###"
exit 0
