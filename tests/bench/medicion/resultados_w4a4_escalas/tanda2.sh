#!/bin/bash
R=/home/usuario/Proyectos/genesis-vllm-patches/tests/bench/medicion/resultados_w4a4_escalas
RR=/home/usuario/Proyectos/genesis-vllm-patches/tests/bench/medicion/resultados_rot
CD=/home/usuario/Proyectos/genesis-vllm-patches/compose
CN=genesis-27b-qwen38-ar4ikov-awq
A="Authorization: Bearer <REDACTADO: clave rotada 2026-09-19>"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $R/estado.log; }
until grep -q "fin tanda" $RR/estado.log 2>/dev/null; do sleep 60; done
bajar(){ for f in $CD/docker-compose.*.yml; do grep -q "container_name: $CN" $f && docker compose -f $f down >/dev/null 2>&1; done; docker rm -f $CN >/dev/null 2>&1; }
arrancar(){ bajar
  docker run --rm -v /home/usuario/.cache/vllm:/c alpine rm -rf /c/qwen38-27b-ar4ikov-awq/torch_compile_cache
  docker compose -f $1 up -d >/dev/null 2>&1; sleep 60
  for i in $(seq 1 200); do curl -s -m 5 -H "$A" 127.0.0.1:8320/v1/models | grep -q '"id"' && return 0; docker ps | grep -q $CN || return 1; sleep 10; done; return 1; }
log "inicio tanda escalas W4A4 gate_up"
for v in wush-g256 wush-g0 had-g256 had-g0; do
  T=sim_${v//-/_}
  if arrancar $CD/docker-compose.sim-$v.yml; then
    log "$T arriba"; mkdir -p $R/$T; cd /tmp/g115
    timeout 900 python3 sanidad.py > $R/$T/sanidad.txt 2>&1
    timeout 1800 python3 pocas.py $T > $R/$T/pocas.txt 2>&1
    timeout 3600 python3 sondeos.py $T 160 > $R/$T/sondeos.txt 2>&1
    log "$T: $(grep -h '/6' $R/$T/pocas.txt) | $(grep -o 'top-1.*' $R/$T/sondeos.txt)"
  else log "$T NO ARRANCO"; docker logs --tail 150 $CN > $R/${T}_fallo.log 2>&1; fi
done
cd /tmp/g115 && python3 sondeos.py --comparar fp16 wush_gate_up sim_wush_g256 sim_wush_g0 sim_had_g256 sim_had_g0 > $R/comparacion.txt 2>&1
bajar; docker compose -f $CD/docker-compose.qwen38-27b-noon-w4a8.yml up -d >/dev/null 2>&1
log "fin tanda escalas (queda arriba noon W4A8 + PN125)"
