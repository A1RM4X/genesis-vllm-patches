#!/bin/bash
R=/home/usuario/Proyectos/genesis-vllm-patches/tests/bench/medicion/resultados_tres
CD=/home/usuario/Proyectos/genesis-vllm-patches/compose
CN=genesis-27b-qwen38-ar4ikov-awq
A="Authorization: Bearer <REDACTADO: clave rotada 2026-09-19>"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $R/estado.log; }
bajar(){ for f in $CD/docker-compose.*.yml; do grep -q "container_name: $CN" $f && docker compose -f $f down >/dev/null 2>&1; done; docker rm -f $CN >/dev/null 2>&1; }
arrancar(){ bajar
  docker run --rm -v /home/usuario/.cache/vllm:/c alpine rm -rf /c/$2/torch_compile_cache
  docker compose -f $1 up -d >/dev/null 2>&1; sleep 60
  for i in $(seq 1 200); do curl -s -m 5 -H "$A" 127.0.0.1:8320/v1/models | grep -q '"id"' && return 0; docker ps | grep -q $CN || return 1; sleep 10; done; return 1; }
log "inicio: 1) escalas en un solo lado"
for v in had-w256-a0 had-w0-a256; do
  T=sim_${v//-/_}; mkdir -p $R/$T
  if arrancar $CD/docker-compose.sim-$v.yml qwen38-27b-ar4ikov-awq; then
    cd /tmp/g115; timeout 900 python3 sanidad.py > $R/$T/sanidad.txt 2>&1
    timeout 1800 python3 pocas.py $T > $R/$T/pocas.txt 2>&1
    timeout 3600 python3 sondeos.py $T 160 > $R/$T/sondeos.txt 2>&1
    log "$T: $(cat $R/$T/pocas.txt) | $(grep -o 'logp medio.*' $R/$T/sondeos.txt)"
  else log "$T NO ARRANCO"; docker logs --tail 150 $CN > $R/${T}_fallo.log 2>&1; fi
done
cd /tmp/g115 && python3 sondeos.py --comparar fp16 sim_had_g256 sim_had_g0 sim_had_w256_a0 sim_had_w0_a256 > $R/comparacion_un_lado.txt 2>&1
log "2) noon W4A8 + PN125: curva PP vs largo + estabilidad"
if arrancar $CD/docker-compose.qwen38-27b-noon-w4a8.yml qwen38-27b-noon-w4a8; then
  cd /tmp/g115; timeout 900 python3 sanidad.py > $R/noon_sanidad.txt 2>&1
  for reps in 90 178 356 711 1422 2222; do
    echo "== reps $reps" >> $R/noon_pp_curva.txt
    timeout 1800 python3 pp.py curva$reps $reps >> $R/noon_pp_curva.txt 2>&1
  done
  log "curva PP: $(grep 'corrida 2' $R/noon_pp_curva.txt | tr '\n' ' ')"
  timeout 1800 python3 largo.py noon_final > $R/noon_largo.txt 2>&1
  timeout 1800 python3 tar_multi.py noon_final > $R/noon_tar.txt 2>&1
  timeout 1800 python3 bordes.py > $R/noon_bordes.txt 2>&1
  timeout 1800 python3 bloques.py > $R/noon_bloques.txt 2>&1
  timeout 1800 python3 suite_calidad.py > $R/noon_suite.txt 2>&1
  log "estabilidad: $(tail -1 $R/noon_largo.txt) | $(grep TOTAL $R/noon_tar.txt) | bordes: $(tail -2 $R/noon_bordes.txt | tr '\n' ' ') | bloques: $(tail -2 $R/noon_bloques.txt | tr '\n' ' ') | $(grep aciertos $R/noon_suite.txt)"
else log "noon NO ARRANCO"; docker logs --tail 150 $CN > $R/noon_fallo.log 2>&1; fi
log "fin (queda arriba noon W4A8 + PN125 para opencode)"
