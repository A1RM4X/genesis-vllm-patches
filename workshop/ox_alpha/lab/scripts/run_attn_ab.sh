#!/usr/bin/env bash
# A/B del backend de atención sobre el modelo REAL (27B FP8 TP=2 MTP K=3):
#   FLASHINFER (producción actual) vs TRITON_ATTN (alternativa sm_86 con KV fp8).
# Dos procesos separados (el backend se elige al construir el engine).
set -uo pipefail

for BK in FLASHINFER TRITON_ATTN; do
  echo ""
  echo "########## ATTENTION A/B: $BK ##########"
  ATTN_BACKEND=$BK python3 /lab/scripts/trace_real.py 2>&1 \
    | tee "/lab/results/attn_ab_${BK}.log" | grep -E "GEN_DONE|SAMPLE_TEXT|backend|rejected"
done
echo "### ATTN A/B DONE ###"
exit 0
