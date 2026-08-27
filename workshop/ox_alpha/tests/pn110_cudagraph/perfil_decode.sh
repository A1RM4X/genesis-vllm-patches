#!/usr/bin/env bash
# perfil_decode.sh — perfil por kernel de un paso de DECODE real, con cudagraphs.
#
# Usa el profiler integrado de vLLM (VLLM_TORCH_PROFILER_DIR + /start_profile).
# Uso: ./perfil_decode.sh <label> <PN110_ENABLE> <PN110_SK> [HYBRID] [SWAP_ONLY_SK]
set -uo pipefail
cd "$(dirname "$0")"
LABEL=$1; EN=$2; SK=$3; HYB=${4:-0}; SWAP=${5:-1}
COMPOSE=docker-compose.pn110-ab.yml
DIS=0; [ "$EN" = "0" ] && DIS=1
mkdir -p traces/$LABEL

docker compose -f $COMPOSE down >/dev/null 2>&1
PN110_ENABLE=$EN PN110_DISABLE=$DIS PN110_SK=$SK PN110_HYBRID=$HYB \
  GENESIS_PN110_SWAP_ONLY_SK=$SWAP VLLM_TORCH_PROFILER_DIR=/traces/$LABEL \
  docker compose -f $COMPOSE up -d >/dev/null 2>&1

echo "[$LABEL] esperando health..."
for i in $(seq 1 60); do
  curl -s -m 3 -o /dev/null http://127.0.0.1:8390/health 2>/dev/null && break
  docker ps --filter name=genesis-pn110-cg --format x | grep -q x || { echo "[$LABEL] contenedor murio"; exit 1; }
  sleep 15
done
curl -s -m 3 -o /dev/null http://127.0.0.1:8390/health || { echo "[$LABEL] timeout"; exit 1; }

# calentar (JIT + cudagraphs ya capturados en el boot, esto llena el camino)
curl -s -m 180 http://127.0.0.1:8390/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8","prompt":"hola","max_tokens":32,"temperature":0}' >/dev/null

echo "[$LABEL] profiling..."
echo "  start_profile -> $(curl -s -m 30 -w ' HTTP:%{http_code}' -X POST http://127.0.0.1:8390/start_profile 2>&1 | tail -c 200)"
curl -s -m 180 http://127.0.0.1:8390/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8","prompt":"Contá del uno al treinta separado por comas:","max_tokens":64,"temperature":0}' >/dev/null
echo "  stop_profile  -> $(curl -s -m 180 -w ' HTTP:%{http_code}' -X POST http://127.0.0.1:8390/stop_profile 2>&1 | tail -c 200)"
echo "  esperando volcado de la traza..."
for i in $(seq 1 24); do
  n=$(docker exec genesis-pn110-cg sh -c "ls /traces/$LABEL 2>/dev/null | wc -l" 2>/dev/null || echo 0)
  [ "$n" -gt 0 ] && { echo "  traza escrita ($n archivos)"; sleep 10; break; }
  sleep 10
done
docker compose -f $COMPOSE down >/dev/null 2>&1
echo "[$LABEL] traces:"; ls -la traces/$LABEL 2>/dev/null | tail -4
