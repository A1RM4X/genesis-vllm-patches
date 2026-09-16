#!/usr/bin/env bash
# Corrida SMALL: modelo chico descargado de HF, TP=1, EngineCore in-process
# (VLLM_ENABLE_V1_MULTIPROCESSING=0) para visibilidad total del profiler.
set -uo pipefail

echo "[run_small] pid=$$ LAB_MODEL=${LAB_MODEL:-Qwen/Qwen3-0.6B}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"

python3 /lab/scripts/trace_small.py 2>&1 | tee /lab/results/run_small_console.log
RC=${PIPESTATUS[0]}
echo "[run_small] trace_small exit=${RC}"

# Contabilidad de memcpys sobre todas las trazas generadas
python3 /lab/scripts/summarize_trace.py \
  /lab/results/worker_trace_*.json \
  > /lab/results/memcpy_summary_small.txt 2>&1 || true
cat /lab/results/memcpy_summary_small.txt || true
exit 0
