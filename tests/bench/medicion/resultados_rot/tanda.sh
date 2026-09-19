#!/bin/bash
R=/home/usuario/Proyectos/genesis-vllm-patches/tests/bench/medicion/resultados_rot
W8=/home/usuario/Proyectos/genesis-vllm-patches/tests/bench/medicion/resultados_w4a8
CD=/home/usuario/Proyectos/genesis-vllm-patches/compose
CN=genesis-27b-qwen38-ar4ikov-awq
A="Authorization: Bearer ${VLLM_API_KEY:?falta VLLM_API_KEY}"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $R/estado.log; }
bajar(){ for f in $CD/docker-compose.*.yml; do grep -q "container_name: $CN" $f && docker compose -f $f down >/dev/null 2>&1; done; docker rm -f $CN >/dev/null 2>&1; }
arrancar(){ bajar
  docker run --rm -v /home/usuario/.cache/vllm:/c alpine rm -rf /c/qwen38-27b-noon-w4a8/torch_compile_cache
  docker compose -f $1 up -d >/dev/null 2>&1; sleep 60
  for i in $(seq 1 200); do curl -s -m 5 -H "$A" 127.0.0.1:8320/v1/models | grep -q '"id"' && return 0; docker ps | grep -q $CN || return 1; sleep 10; done; return 1; }
medir(){ T=$1; O=$R/$T; mkdir -p $O
  docker logs $CN > $O/arranque.log 2>&1
  grep -E "KV cache size|PN126|PN125\]" $O/arranque.log | sort | uniq -c | tail -8 > $O/resumen_arranque.txt
  cd /tmp/g115
  timeout 900 python3 sanidad.py > $O/sanidad.txt 2>&1
  timeout 1800 python3 pp.py $T > $O/pp.txt 2>&1
  timeout 1800 python3 largo.py $T > $O/largo.txt 2>&1
  timeout 1800 python3 tar_multi.py $T > $O/tar.txt 2>&1
  timeout 1800 python3 pocas.py $T > $O/pocas.txt 2>&1
  timeout 3600 python3 suite_calidad.py > $O/suite.txt 2>&1
  timeout 3600 python3 sondeos.py $T 160 > $O/sondeos.txt 2>&1
  log "$T: $(tail -1 $O/pp.txt) | $(tail -1 $O/largo.txt) | $(grep TOTAL $O/tar.txt) | $(grep -h '/6\|aciertos' $O/pocas.txt $O/suite.txt | tr '\n' ' ') | $(grep -o 'logp medio.*' $O/sondeos.txt)"; }
correr(){ if arrancar $CD/docker-compose.rot-$2.yml; then log "$1 arriba"; medir $1; else log "$1 NO ARRANCO"; docker logs --tail 150 $CN > $R/$1_fallo.log 2>&1; fi; }

log "inicio tanda de rotacion q/k"
# 1) captura de gramianos para WUSH (eager, KV fp8)
if arrancar $CD/docker-compose.rot-captura.yml; then
  log "captura arriba"
  D=/home/usuario/.claude/jobs/fc6ace03/tmp python3 /home/usuario/.claude/jobs/fc6ace03/tmp/calib.py >> $R/captura.log 2>&1
  docker exec $CN touch /dev/shm/pn126_volcar
  cd /tmp/g115 && timeout 900 python3 sanidad.py >> $R/captura.log 2>&1
  sleep 5; docker logs $CN 2>&1 | grep "PN126" | tail -4 >> $R/estado.log
  ls -la /home/usuario/Proyectos/kv-offload/pn126_wush >> $R/estado.log 2>&1
else log "captura NO ARRANCO"; docker logs --tail 150 $CN > $R/captura_fallo.log 2>&1; fi
correr rot_i8_had i8-had
correr rot_i8_wush i8-wush
correr rot_fp8_had fp8-had
correr rot_fp8_wush fp8-wush
correr rot_fp8_control fp8-control
cd /tmp/g115
cp $W8/../resultados_w4a8/noon_w4a8_pn125/pp.txt /dev/null 2>&1
python3 sondeos.py --comparar noon_w4a8_pn125 rot_fp8_control rot_fp8_had rot_fp8_wush noon_w4a8_kvint8 rot_i8_had rot_i8_wush noon_w4a16 > $R/sondeos_vs_w4a8fp8.txt 2>&1
python3 sondeos.py --comparar noon_w4a16 noon_w4a8_pn125 rot_fp8_control rot_fp8_had rot_fp8_wush noon_w4a8_kvint8 rot_i8_had rot_i8_wush > $R/sondeos_vs_w4a16.txt 2>&1
bajar; docker compose -f $CD/docker-compose.qwen38-27b-noon-w4a8.yml up -d >/dev/null 2>&1
log "fin tanda (queda arriba noon W4A8 + PN125, KV fp8)"
