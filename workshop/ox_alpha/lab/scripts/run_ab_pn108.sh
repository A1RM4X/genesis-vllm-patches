#!/usr/bin/env bash
# CK-1.1 — A/B de aceptancia MTP: baseline vs PN108 (draft lm_head FP8).
# Métricas: GEN_DONE tps + líneas de aceptancia del logger periódico.
set -uo pipefail
mkdir -p /lab/results/ab_pn108

run_one() {
  local TAG=$1; shift
  echo ""
  echo "########## A/B: $TAG ##########"
  env "$@" LAB_LOG_STATS=1 LAB_PROMPTS=10 LAB_TOKENS=250 \
    python3 /lab/scripts/trace_real.py 2>&1 \
    | tee "/lab/results/ab_pn108/${TAG}.log" \
    | grep -iE "gen_done|SpecDecoding metrics|PN108" || true
}

run_one baseline env -u GENESIS_ENABLE_PN108_DRAFT_FP8_LM_HEAD
run_one pn108   GENESIS_ENABLE_PN108_DRAFT_FP8_LM_HEAD=1 LAB_ENABLE_PN108=1 LAB_DUMP_MODEL=1

echo ""; echo "### A/B PN108 DONE ###"
exit 0
