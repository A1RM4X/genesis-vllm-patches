#!/bin/bash
# Arranca el compose v029 en ciclo hasta atrapar un arranque ROTO, y lo deja levantado.
#
# El sintoma conocido: /v1/chat/completions devuelve 2 tokens con content vacio. Se sonda con
# TRES pedidos distintos para separar "el modelo esta roto" de "el template/parser esta roto":
#   A) chat normal (pasa por el template, mete los tokens especiales del final del vocabulario)
#   B) los MISMOS tokens del template, por /v1/completions (sin parser de reasoning en el medio)
#   C) texto plano por /v1/completions (sin un solo token especial)
# Si A y B fallan y C anda, el dano esta en las ultimas filas del vocabulario.
set -u
cd /home/usuario/Proyectos/genesis-vllm-patches/compose
YML=docker-compose.qwen38-27b-noon-w4a8-v029.yml
NOMBRE=genesis-27b-v029
IP=172.20.0.228:8320
K=<REDACTADO: clave rotada 2026-09-19>
N=${1:-10}

pedir() {  # $1 = cuerpo json, $2 = ruta
  curl -s -m 180 "http://$IP$2" -H 'Content-Type: application/json' \
    -H "Authorization: Bearer $K" -d "$1" | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin); print(d['usage']['completion_tokens'])
except Exception: print('ERR')" 2>/dev/null
}

A='{"model":"qwen3.8","max_tokens":60,"temperature":0,"messages":[{"role":"user","content":"Contame en dos frases que es Paris."}]}'
B='{"model":"qwen3.8","prompt":[248045,846,198,65717,248046,198,248045,74455,198,248068,198],"max_tokens":60,"temperature":0}'
C='{"model":"qwen3.8","prompt":"La capital de Francia es","max_tokens":60,"temperature":0}'

for n in $(seq 1 "$N"); do
  docker compose -f $YML down >/dev/null 2>&1
  find /dev/shm -maxdepth 1 -type f \( -name "*vllm*" -o -name "*offload*" -o -name "psm*" \) -delete 2>/dev/null
  docker compose -f $YML up -d >/dev/null 2>&1
  arriba=no
  for i in $(seq 1 50); do
    curl -sf -H "Authorization: Bearer $K" -m 5 "http://$IP/health" >/dev/null 2>&1 && { arriba=si; break; }
    docker logs $NOMBRE 2>&1 | grep -q "Engine core initialization failed" && break
    sleep 20
  done
  if [ "$arriba" = no ]; then echo "arranque $n: NO LEVANTO"; continue; fi
  a=$(pedir "$A" /v1/chat/completions); b=$(pedir "$B" /v1/completions); c=$(pedir "$C" /v1/completions)
  kv=$(docker logs $NOMBRE 2>&1 | grep -o "GPU KV cache size: [0-9,]*" | tail -1)
  if [ "$a" = "60" ] && [ "$b" = "60" ] && [ "$c" = "60" ]; then
    echo "arranque $n: OK    (A=$a B=$b C=$c) | $kv"
  else
    echo "arranque $n: ROTO  (A=$a B=$b C=$c) | $kv"
    echo ">>> lo dejo levantado para diagnosticar en caliente"
    exit 7
  fi
done
echo "== $N arranques, ninguno roto =="
