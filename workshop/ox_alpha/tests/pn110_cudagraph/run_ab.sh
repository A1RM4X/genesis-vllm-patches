#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# run_ab.sh — sweep de debug de lentitud PN110 (ir desactivando cosas).
#
# Reproduce el escenario del usuario (modelo orcarouter/Qwen3.8-27B-Uncensored-FP8,
# TP=2, MTP K=3, cudagraphs ON) y mide performance bajo distintas configuraciones
# de PN110 para aislar DÓNDE está la lentitud:
#
#   off  : PN110 entero OFF  (GENESIS_DISABLE_PN110=1)        -> baseline FP8/Marlin
#   sk0  : PN110 ON, superkernels OFF (GENESIS_PN110_SK=0)    -> solo boilerplate INT8
#   sk1  : PN110 ON, superkernels ON  (GENESIS_PN110_SK=1)    -> estado actual "lento"
#
# Para cada config mide (bench_pn110.py): TTFT, decode tok/s, prefill tok/s, y
# si los CUDA graphs se capturaron o fallaron (fallo => modo eager ~2x lento).
#
# Uso:
#   ./run_ab.sh            # sweep completo: off + sk0 + sk1
#   ./run_ab.sh sk0 sk1    # solo comparar boilerplate vs full
#   ./run_ab.sh off        # solo baseline
set -uo pipefail
cd "$(dirname "$0")"

COMPOSE=${COMPOSE_AB:-docker-compose.pn110-cg.yml}
CONT=genesis-pn110-cg
OUTDIR=/tmp/pn110_bench
mkdir -p "$OUTDIR"

echo "=== [0] Liberando GPUs: stop de contenedores con dispositivo GPU ==="
for c in $(docker ps --format '{{.Names}}'); do
  gpus=$(docker inspect --format '{{range .HostConfig.DeviceRequests}}{{range .Capabilities}}{{.}}{{end}}{{end}}' "$c" 2>/dev/null)
  if [ -n "$gpus" ]; then
    echo "  stop $c (usa GPU)"; docker stop "$c" >/dev/null 2>&1 || true
  fi
done

run_one() {
  local TAG=$1; shift   # TAG luego de las env vars restantes
  echo ""
  echo "########## PN110 debug: $TAG ##########"
  docker compose -f "$COMPOSE" down >/dev/null 2>&1 || true
  # $@ son las env vars exportadas antes de llamar
  docker compose -f "$COMPOSE" up -d
  sleep 5
  local json="$OUTDIR/report_${TAG}.json"
  # Borrar el reporte previo ANTES de medir. Sin esto, si el arranque falla,
  # bench_pn110.py no escribe nada y la comparativa de abajo imprime el JSON de
  # una corrida ANTERIOR como si fuera de esta -- paso de verdad: se reporto
  # sk1=97.89 tok/s de una corrida vieja cuando el server ni habia arrancado.
  rm -f "$json"
  python3 bench_pn110.py --container "$CONT" --port 8390 --label "$TAG" \
    --prompt-decode-len 24 --gen-decode 200 --prompt-prefill-len 2000 \
    --gen-prefill 16 --requests 3 --warmup 1 --timeout 480 --req-timeout 180 \
    --json-out "$json"
  echo "  -> reporte: $json"
  docker compose -f "$COMPOSE" down >/dev/null 2>&1 || true
}

# Configuraciones: TAG + env vars (exportadas en el subshell de run_one)
CONFIGS=()
if [ $# -eq 0 ]; then CONFIGS=(off sk0 sk1); else CONFIGS=("$@"); fi

for cfg in "${CONFIGS[@]}"; do
  case "$cfg" in
    off)
      PN110_ENABLE=0 PN110_DISABLE=1 PN110_SK=0 run_one off ;;
    sk0)
      PN110_ENABLE=1 PN110_DISABLE=0 PN110_SK=0 run_one sk0 ;;
    sk1)
      PN110_ENABLE=1 PN110_DISABLE=0 PN110_SK=1 run_one sk1 ;;
    *)
      echo "config desconocida: $cfg (usar off|sk0|sk1)";;
  esac
done

echo ""
echo "==================== COMPARATIVA PN110 (decode tok/s | TTFT s | cudagraphs) ===================="
printf "%-10s | %12s | %10s | %12s | %10s | %s\n" "CONFIG" "decode_tok/s" "TTFT_s" "prefill_tok/s" "cudagraphs" "cg_error"
for cfg in "${CONFIGS[@]}"; do
  j="$OUTDIR/report_${cfg}.json"
  [ -f "$j" ] || { printf "%-10s | %s\n" "$cfg" "SIN DATO (el server no arranco o el bench fallo)"; continue; }
  python3 - "$j" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
def f(x): return "%.2f"%x if isinstance(x,(int,float)) else str(x)
print("%-10s | %12s | %10s | %12s | %10s | %s" % (
  r.get("label",""), f(r.get("decode_tok_s")), f(r.get("ttft_s")),
  f(r.get("prefill_tok_s")), str(r.get("cudagraphs_captured")), str(r.get("cudagraph_error"))))
PY
done
echo "==========================================================================================================="
echo "Lectura: si sk0 (boilerplate) es RÁPIDO y sk1 (full) es LENTO -> la lentitud está en los superkernels."
echo "         si sk0 también es LENTO vs off -> la lentitud está en el boilerplate INT8 de PN110."
echo "         si cg_error=True en sk1 -> los superkernels rompen la captura de cudagraphs (caída a eager)."
