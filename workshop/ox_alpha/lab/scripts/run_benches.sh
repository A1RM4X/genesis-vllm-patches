#!/usr/bin/env bash
# Batería completa de micro-benchmarks de kernels (sin cargar modelo).
set -uo pipefail
mkdir -p /lab/results/benches

run() {
  echo ""
  echo "########## $1 ##########"
  shift
  env "$@" python3 "/lab/scripts/$SCRIPT" 2>&1 | tee -a "/lab/results/benches/${SCRIPT%.py}.log"
}

SCRIPT=bench_gemm.py;   run "GEMM shootout"
SCRIPT=bench_sampler.py; run "Sampler"

# FLA: 4 combinaciones precision × bkv
SCRIPT=bench_fla.py
FLA_TRIL_PRECISION=ieee LAB_EXPAND_BKV=0 run "FLA ieee/default"
FLA_TRIL_PRECISION=tf32 LAB_EXPAND_BKV=0 run "FLA tf32/default"
FLA_TRIL_PRECISION=ieee LAB_EXPAND_BKV=1 run "FLA ieee/BKV-ampliado"
FLA_TRIL_PRECISION=tf32 LAB_EXPAND_BKV=1 run "FLA tf32/BKV-ampliado"

SCRIPT=bench_memcpy.py; run "Memcpy estados mamba"
echo ""; echo "### BENCHES DONE ###"
exit 0
