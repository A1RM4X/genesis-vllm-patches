#!/bin/bash
R=/home/usuario/Proyectos/genesis-vllm-patches/tests/bench/medicion/resultados_w4a4_escalas
RR=/home/usuario/Proyectos/genesis-vllm-patches/tests/bench/medicion/resultados_rot
CD=/home/usuario/Proyectos/genesis-vllm-patches/compose
CN=genesis-27b-qwen38-ar4ikov-awq
A="Authorization: Bearer ${VLLM_API_KEY:?falta VLLM_API_KEY}"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $R/estado.log; }

bajar(){ for f in $CD/docker-compose.*.yml; do grep -q "container_name: $CN" $f && docker compose -f $f down >/dev/null 2>&1; done; docker rm -f $CN >/dev/null 2>&1; }
arrancar(){ bajar
  docker run --rm -v /home/usuario/.cache/vllm:/c alpine rm -rf /c/qwen38-27b-ar4ikov-awq/torch_compile_cache
  docker compose -f $1 up -d >/dev/null 2>&1; sleep 60
  for i in $(seq 1 200); do curl -s -m 5 -H "$A" 127.0.0.1:8320/v1/models | grep -q '"id"' && return 0; docker ps | grep -q $CN || return 1; sleep 10; done; return 1; }
until grep -q "fin tanda 2" $R/estado.log; do sleep 30; done; log "inicio tanda 3: densa"
for v in densa-g0; do
  T=sim_${v//-/_}
  if arrancar $CD/docker-compose.sim-$v.yml; then
    log "$T arriba"; mkdir -p $R/$T; cd /tmp/g115
    timeout 900 python3 sanidad.py > $R/$T/sanidad.txt 2>&1
    timeout 1800 python3 pocas.py $T > $R/$T/pocas.txt 2>&1
    timeout 3600 python3 sondeos.py $T 160 > $R/$T/sondeos.txt 2>&1
    log "$T: $(grep -h '/6' $R/$T/pocas.txt) | $(grep -o 'top-1.*' $R/$T/sondeos.txt)"
  else log "$T NO ARRANCO"; docker logs --tail 150 $CN > $R/${T}_fallo.log 2>&1; fi
done
cd /tmp/g115 && python3 sondeos.py --comparar fp16 wush_gate_up sim_had_g256 sim_had_g0 sim_densa_g0 sim_had8192_g0 sim_had_g1024 > $R/comparacion3.txt 2>&1
bajar; docker compose -f $CD/docker-compose.qwen38-27b-noon-w4a8.yml up -d >/dev/null 2>&1
log "fin tanda 3 (queda arriba noon W4A8 + PN125)"
