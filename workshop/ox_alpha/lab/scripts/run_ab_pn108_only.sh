#!/usr/bin/env bash
set -uo pipefail
mkdir -p /lab/results/ab_pn108
echo "########## A/B: pn108 (retry con activacion real) ##########"
GENESIS_ENABLE_PN108_DRAFT_FP8_LM_HEAD=1 LAB_ENABLE_PN108=1 LAB_DUMP_MODEL=1 LAB_LOG_STATS=1 LAB_PROMPTS=10 LAB_TOKENS=250 python3 /lab/scripts/trace_real.py 2>&1 | tee /lab/results/ab_pn108/pn108.log | grep -iE "gen_done|SpecDecoding metrics|PN108" || true
echo AB-PN108-DONE
