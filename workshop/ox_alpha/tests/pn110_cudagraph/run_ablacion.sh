#!/usr/bin/env bash
# run_ablacion.sh — ablación fina de PN110: por régimen (B/C) y por super kernel.
#
# A diferencia de run_ab.sh (que sólo prende/apaga TODO), acá se controla:
#   PN110_HYBRID   0 = Diseño B per-channel (alternativo: cutlass_scaled_mm)
#                  1 = Diseño C diádico     (alternativo: int8_hybrid_gemm)
#   PN110_SK       0/1
#   SK_ONLY        lista blanca, p.ej. "SK-05,SK-06"  -> sólo esos van por super kernel
#   SK_SKIP        lista negra
#
# Uso:
#   ./run_ablacion.sh <label> <HYBRID> <SK> [SK_ONLY] [SK_SKIP]
# Ej:
#   ./run_ablacion.sh b0_cutlass 0 0
#   ./run_ablacion.sh b1_todos   0 1
#   ./run_ablacion.sh b1_solo05  0 1 SK-05
set -uo pipefail
cd "$(dirname "$0")"
# Se usa el compose por --gpu-memory-utilization (no el de --kv-cache-memory
# fijo): el KV explicito esta calibrado para los pesos del BASELINE (14.43
# GiB/GPU) y con PN110 los pesos son 15.89, asi que 7.11 GiB de KV no entra
# y el arranque muere con OOM. El util se adapta a lo que pese cada config.
COMPOSE=${COMPOSE_ABL:-docker-compose.pn110-ab.yml}
CONT=genesis-pn110-cg
OUT=/tmp/pn110_bench/ablacion
mkdir -p "$OUT"

LABEL=$1; HYB=$2; SK=$3; ONLY=${4:-}; SKIP=${5:-}; SWAP=${6:-1}

echo "########## $LABEL  (HYBRID=$HYB SK=$SK ONLY='$ONLY' SKIP='$SKIP' SWAP_ONLY_SK=$SWAP) ##########"
docker compose -f "$COMPOSE" down >/dev/null 2>&1
PN110_ENABLE=1 PN110_DISABLE=0 PN110_SK=$SK PN110_HYBRID=$HYB \
  GENESIS_PN110_SK_ONLY="$ONLY" GENESIS_PN110_SK_SKIP="$SKIP" \
  GENESIS_PN110_SWAP_ONLY_SK="$SWAP" \
  docker compose -f "$COMPOSE" up -d >/dev/null 2>&1
sleep 5
python3 bench_pn110.py --container "$CONT" --port 8390 --label "$LABEL" \
  --prompt-decode-len 24 --gen-decode 200 --prompt-prefill-len 2000 \
  --gen-prefill 16 --requests 3 --warmup 1 --timeout 600 --req-timeout 180 \
  --json-out "$OUT/report_${LABEL}.json"
# capturar tambien el perfil de VRAM y las capas despachadas
docker logs "$CONT" 2>&1 | grep "gpu_worker.py:816" | head -1 | sed 's/^(\S* pid=[0-9]*) //' > "$OUT/vram_${LABEL}.txt"
docker logs "$CONT" 2>&1 | grep -E "capas por super kernel ACTIVO|DESACTIVADOS" | tail -2 | sed 's/^(\S* pid=[0-9]*) //' >> "$OUT/vram_${LABEL}.txt"
docker compose -f "$COMPOSE" down >/dev/null 2>&1
echo "  -> $OUT/report_${LABEL}.json"
