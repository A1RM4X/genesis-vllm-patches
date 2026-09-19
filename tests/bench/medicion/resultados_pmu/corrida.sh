#!/bin/bash
R=/home/usuario/Proyectos/genesis-vllm-patches/tests/bench/medicion/resultados_pmu
CD=/home/usuario/Proyectos/genesis-vllm-patches/compose
M=/home/usuario/Proyectos/genesis-vllm-patches/tests/bench/medicion
CN=genesis-27b-qwen38-ar4ikov-awq
A="Authorization: Bearer ${VLLM_API_KEY:?falta VLLM_API_KEY}"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $R/estado.log; }
arrancar(){ docker compose -f $CD/docker-compose.qwen38-27b-noon-w4a8.yml down >/dev/null 2>&1; docker compose -f $CD/docker-compose.pmu16-noon-w4a8.yml down >/dev/null 2>&1
  docker run --rm -v /home/usuario/.cache/vllm:/c alpine rm -rf /c/qwen38-27b-noon-w4a8/torch_compile_cache
  docker compose -f $1 up -d >/dev/null 2>&1; sleep 60
  for i in $(seq 1 200); do curl -s -m 5 -H "$A" 127.0.0.1:8320/v1/models | grep -q '"id"' && return 0; docker ps | grep -q $CN || return 1; sleep 10; done; return 1; }
for rep in 1 2; do
 for v in base pmu16; do
  f=$CD/docker-compose.qwen38-27b-noon-w4a8.yml; [ $v = pmu16 ] && f=$CD/docker-compose.pmu16-noon-w4a8.yml
  if arrancar $f; then
    cd /tmp/g115; timeout 900 python3 sanidad.py >/dev/null 2>&1
    timeout 3600 python3 $M/turnos.py ${v}_r$rep 12 > $R/${v}_r$rep.txt 2>&1
    [ $rep = 1 ] && timeout 1800 python3 pocas.py ${v} > $R/${v}_pocas.txt 2>&1 && timeout 1800 python3 pp.py ${v} > $R/${v}_pp.txt 2>&1
    log "$v r$rep: $(tail -1 $R/${v}_r$rep.txt) | $(cat $R/${v}_pocas.txt 2>/dev/null | head -1) | $(tail -1 $R/${v}_pp.txt 2>/dev/null)"
  else log "$v NO ARRANCO"; docker logs --tail 150 $CN > $R/${v}_fallo.log 2>&1; fi
 done
done
docker compose -f $CD/docker-compose.pmu16-noon-w4a8.yml down >/dev/null 2>&1
docker run --rm -v /home/usuario/.cache/vllm:/c alpine rm -rf /c/qwen38-27b-noon-w4a8/torch_compile_cache
docker compose -f $CD/docker-compose.qwen38-27b-noon-w4a8.yml up -d >/dev/null 2>&1
log "fin (queda arriba noon W4A8 + PN130 sin prefix-match-unit)"
