#!/usr/bin/env bash
# perfil_decode.sh — perfil por kernel de un paso de DECODE real, con cudagraphs.
#
# Usa el profiler integrado de vLLM v0.27.1: --profiler-config + /start_profile.
#
# OJO: VLLM_TORCH_PROFILER_DIR fue eliminada en v0.27.1. Las rutas del profiler
# se montan en entrypoints/serve/profile/api_router.py:attach_router() SOLO si
# args.profiler_config.profiler is not None. Con la variable vieja el endpoint
# devolvia 404 y la traza quedaba vacia sin ningun error visible.
#
# delay_iterations salta el prefill; max_iterations acota a N pasos de decode,
# asi la traza es de decode puro y no mezcla las dos fases.
# Uso: ./perfil_decode.sh <label> <PN110_ENABLE> <PN110_SK> [HYBRID] [SWAP_ONLY_SK]
set -uo pipefail
cd "$(dirname "$0")"
LABEL=$1; EN=$2; SK=$3; HYB=${4:-0}; SWAP=${5:-1}
COMPOSE=${COMPOSE_PERFIL:-docker-compose.pn110-cg.yml}
DIS=0; [ "$EN" = "0" ] && DIS=1
mkdir -p traces/$LABEL

docker compose -f $COMPOSE down >/dev/null 2>&1
PN110_ENABLE=$EN PN110_DISABLE=$DIS PN110_SK=$SK PN110_HYBRID=$HYB \
  GENESIS_PN110_SWAP_ONLY_SK=$SWAP \
  PROF_KIND='"torch"' PROF_DIR=/traces/$LABEL PROF_DELAY=${PROF_DELAY:-4} PROF_MAX=${PROF_MAX:-40} \
  docker compose -f $COMPOSE up -d >/dev/null 2>&1

echo "[$LABEL] esperando health..."
for i in $(seq 1 60); do
  curl -s -m 3 -o /dev/null http://127.0.0.1:8390/health 2>/dev/null && break
  docker ps --filter name=genesis-pn110-cg --format x | grep -q x || { echo "[$LABEL] contenedor murio"; exit 1; }
  sleep 15
done
curl -s -m 3 -o /dev/null http://127.0.0.1:8390/health || { echo "[$LABEL] timeout"; exit 1; }

# Calentar. El prompt tiene que ser LARGO, y no por gusto:
# flashinfer/prefill.py:2302 fija _max_total_num_rows con el PRIMER plan() en
# modo cudagraph y despues lo congela. Si el primer request es "hola" (~2 filas
# de prefill) el latch queda en 2, y el drafting de MTP K=3 -- que necesita
# 1+3 = 4 filas -- revienta con
#   ValueError: The total number of rows in qo_indptr 4 in cuda graph mode
#   cannot exceed the number of rows set during initialization
# matando el EngineCore antes de llegar a perfilar nada. Verificado: corto
# primero -> muere; largo primero -> los dos requests pasan.
curl -s -m 180 http://127.0.0.1:8390/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8","prompt":"Explicame en detalle y paso a paso como funciona la memoria virtual en un sistema operativo moderno, incluyendo paginacion, TLB y fallos de pagina.","max_tokens":32,"temperature":0}' >/dev/null

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
