#!/bin/bash
# 3 corridas: noon W4A16, noon W4A8, Ar4ikov produccion (re-medida y restaurada)
R=/home/usuario/Proyectos/genesis-vllm-patches/tests/bench/medicion/resultados_w4a8
CD=/home/usuario/Proyectos/genesis-vllm-patches/compose
NOON=$CD/docker-compose.qwen38-27b-noon-w4a8.yml
PROD=$CD/docker-compose.qwen38-27b-ar4ikov-awq.yml
CN=genesis-27b-qwen38-ar4ikov-awq
A="Authorization: Bearer <REDACTADO: clave rotada 2026-09-19>"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $R/estado.log; }

arrancar(){ # $1 compose  $2 cache
  docker compose -f $PROD down >/dev/null 2>&1; docker compose -f $NOON down >/dev/null 2>&1; docker compose -f $CD/docker-compose.qwen38-27b-noon-w4a8-kvint8.yml down >/dev/null 2>&1
  docker run --rm -v /home/usuario/.cache/vllm:/c alpine rm -rf /c/$2/torch_compile_cache
  docker compose -f $1 up -d >/dev/null 2>&1
  sleep 60
  for i in $(seq 1 180); do
    curl -s -m 5 -H "$A" 127.0.0.1:8320/v1/models | grep -q '"id"' && return 0
    docker ps | grep -q $CN || return 1
    sleep 10
  done; return 1
}

medir(){ # $1 tag  $2 incluir_calidad(1/0)
  T=$1; O=$R/$T; mkdir -p $O
  docker logs $CN > $O/arranque.log 2>&1
  grep -E "KV cache size|Maximum concurrency|MarlinLinearKernel|Using .*Kernel|WNA16|WNA8|marlin" $O/arranque.log | sort | uniq -c > $O/kernels_kv.txt
  cd /tmp/g115
  timeout 900  python3 sanidad.py > $O/sanidad.txt 2>&1
  timeout 1800 python3 pp.py $T > $O/pp.txt 2>&1
  timeout 1800 python3 largo.py $T > $O/largo.txt 2>&1
  timeout 1800 python3 tar_multi.py $T > $O/tar.txt 2>&1
  if [ "$2" = 1 ]; then
    timeout 1800 python3 pocas.py $T > $O/pocas.txt 2>&1
    timeout 3600 python3 suite_calidad.py > $O/suite.txt 2>&1
    timeout 3600 python3 sondeos.py $T 160 > $O/sondeos.txt 2>&1
  fi
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv > $O/vram.txt
  log "$T: $(tail -1 $O/pp.txt) | $(tail -1 $O/largo.txt) | $(grep TOTAL $O/tar.txt) | $(grep -h '/6\|aciertos' $O/pocas.txt $O/suite.txt 2>/dev/null | tr '\n' ' ')"
}

KV8=$CD/docker-compose.qwen38-27b-noon-w4a8-kvint8.yml
log "inicio corrida 5 (W4A8 + PN125 + KV int8 Triton/PN124)"
docker compose -f $KV8 down >/dev/null 2>&1
if arrancar $KV8 qwen38-27b-noon-w4a8; then
  log "noon_w4a8_kvint8 arriba"
  medir noon_w4a8_kvint8 1
else log "noon_w4a8_kvint8 NO ARRANCO"; docker logs --tail 150 $CN > $R/noon_w4a8_kvint8_fallo.log 2>&1; fi
cd /tmp/g115 && python3 sondeos.py --comparar noon_w4a8_pn125 noon_w4a8_kvint8 > $R/sondeos_kvint8_vs_kvfp8.txt 2>&1
python3 sondeos.py --comparar noon_w4a16 noon_w4a8_pn125 noon_w4a8_kvint8 > $R/sondeos_kvint8_vs_w4a16.txt 2>&1
log "fin corrida 5"
