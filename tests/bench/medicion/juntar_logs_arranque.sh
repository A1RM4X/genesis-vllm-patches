#!/bin/bash
# Arranca N veces guardando el log de cada arranque, etiquetado OK o ROTO.
cd /home/usuario/Proyectos/genesis-vllm-patches/compose
YML=docker-compose.qwen38-27b-noon-w4a8-v029.yml; K=${VLLM_API_KEY:?falta VLLM_API_KEY}
D=/home/usuario/.claude/jobs/fc6ace03/tmp/logs; mkdir -p $D   # NO borrar: los logs viejos son la referencia
for n in $(seq 1 ${1:-6}); do
  docker compose -f $YML down >/dev/null 2>&1
  find /dev/shm -maxdepth 1 -type f \( -name "*vllm*" -o -name "*offload*" -o -name "psm*" \) -delete 2>/dev/null
  docker compose -f $YML up -d >/dev/null 2>&1
  arriba=no
  for i in $(seq 1 50); do curl -sf -H "Authorization: Bearer $K" -m 5 http://172.20.0.228:8320/health >/dev/null 2>&1 && { arriba=si; break; }; sleep 20; done
  [ "$arriba" = no ] && { echo "$n: NO LEVANTO"; continue; }
  t=$(curl -s -m 180 http://172.20.0.228:8320/v1/completions -H 'Content-Type: application/json' \
      -H "Authorization: Bearer $K" -d '{"model":"qwen3.8","prompt":"La capital de Francia es","max_tokens":40,"temperature":0}' \
      | python3 -c "
import sys,json
try: print(json.load(sys.stdin)['usage']['completion_tokens'])
except Exception: print('ERR')" 2>/dev/null)
  if [ "$t" = "40" ]; then et=OK; else et=ROTO; fi
  docker logs genesis-27b-v029 > $D/${et}_$n.log 2>&1
  echo "$n: $et (tokens=$t)"
done
