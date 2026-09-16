#!/usr/bin/env bash
# Corrida REAL: Qwen3.8-27B-Uncensored-FP8 con TP=2 + MTP K=3, réplica de la
# config PROD (sin KV-offloading para aislar el circuito del token de los
# streams del OffloadingConnector, que se analizan por separado).
set -uo pipefail

echo "[run_real] pid=$$ starting"
python3 /lab/scripts/trace_real.py 2>&1 | tee /lab/results/run_real_console.log
RC=${PIPESTATUS[0]}
echo "[run_real] exit=${RC}"

python3 /lab/scripts/summarize_trace.py \
  /lab/results/worker_trace_*.json \
  > /lab/results/memcpy_summary_real.txt 2>&1 || true
cat /lab/results/memcpy_summary_real.txt || true
exit 0
