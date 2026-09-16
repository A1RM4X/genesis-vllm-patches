#!/usr/bin/env bash
# CK-0.2 — Barrido NCCL_PROTO sobre el modelo real (batch 10).
# Combos: default (lo que elija NCCL) / Ring+LL128 / Ring+Simple.
# Cada combo = boot completo del engine; trazas de worker separadas por combo.
set -uo pipefail
mkdir -p /lab/results/nccl_sweep

for COMBO in default LL128 Simple; do
  echo ""
  echo "########## NCCL COMBO: $COMBO ##########"
  if [ "$COMBO" = "default" ]; then
    env -u NCCL_PROTO -u NCCL_ALGO \
      NCCL_BENCH_PROMPTS=10 NCCL_BENCH_TOKENS=100 \
      python3 /lab/scripts/trace_nccl.py 2>&1 \
      | tee "/lab/results/nccl_sweep/${COMBO}.log" \
      | grep -iE "gen_done|nccl_proto|nccl_algo|warmup done" || true
  else
    NCCL_ALGO=Ring NCCL_PROTO=$COMBO \
      NCCL_BENCH_PROMPTS=10 NCCL_BENCH_TOKENS=100 \
      python3 /lab/scripts/trace_nccl.py 2>&1 \
      | tee "/lab/results/nccl_sweep/${COMBO}.log" \
      | grep -iE "gen_done|nccl_proto|nccl_algo|warmup done" || true
  fi
  # Separar las trazas de ESTE combo y resumirlas
  mkdir -p "/lab/results/nccl_sweep/traces_${COMBO}"
  mv /lab/results/worker_trace_*.json \
     "/lab/results/nccl_sweep/traces_${COMBO}/" 2>/dev/null || true
  python3 /lab/scripts/summarize_trace.py \
    "/lab/results/nccl_sweep/traces_${COMBO}"/worker_trace_*.json \
    > "/lab/results/nccl_sweep/kernels_${COMBO}.txt" 2>&1 || true
  grep -E "kernels\(top\)|DtoH =" "/lab/results/nccl_sweep/kernels_${COMBO}.txt" || true
done
echo "### NCCL SWEEP DONE ###"
exit 0
