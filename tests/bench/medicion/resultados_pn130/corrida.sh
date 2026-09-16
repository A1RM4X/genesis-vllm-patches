#!/bin/bash
R=/home/usuario/Proyectos/genesis-vllm-patches/tests/bench/medicion/resultados_pn130
CD=/home/usuario/Proyectos/genesis-vllm-patches/compose
CN=genesis-27b-qwen38-ar4ikov-awq
A="Authorization: Bearer <REDACTADO: clave rotada 2026-09-19>"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $R/estado.log; }
docker compose -f $CD/docker-compose.qwen38-27b-noon-w4a8.yml down >/dev/null 2>&1
docker run --rm -v /home/usuario/.cache/vllm:/c alpine rm -rf /c/qwen38-27b-noon-w4a8/torch_compile_cache
docker compose -f $CD/docker-compose.qwen38-27b-noon-w4a8.yml up -d >/dev/null 2>&1
sleep 60
for i in $(seq 1 200); do curl -s -m 5 -H "$A" 127.0.0.1:8320/v1/models | grep -q '"id"' && break; docker ps | grep -q $CN || { log "NO ARRANCO"; docker logs --tail 200 $CN > $R/fallo.log 2>&1; exit 1; }; sleep 10; done
docker logs $CN > $R/arranque.log 2>&1
log "arriba: $(grep -E '\[Genesis\] (applied|skipped|failed).*PN1(25|27|28|29|30)' $R/arranque.log | cut -c1-140 | tr '\n' ';')"
cd /tmp/g115
T=noon_w4a8_pn130
timeout 900 python3 sanidad.py > $R/sanidad.txt 2>&1
timeout 1800 python3 pp.py $T > $R/pp.txt 2>&1
timeout 1800 python3 largo.py $T > $R/largo.txt 2>&1
timeout 1800 python3 tar_multi.py $T > $R/tar.txt 2>&1
timeout 1800 python3 pocas.py $T > $R/pocas.txt 2>&1
timeout 3600 python3 suite_calidad.py > $R/suite.txt 2>&1
timeout 3600 python3 sondeos.py $T 160 > $R/sondeos.txt 2>&1
python3 sondeos.py --comparar noon_w4a16 noon_w4a8_pn125 noche_ruido1 noon_w4a8_pn130 > $R/comparacion.txt 2>&1
log "$T: $(tail -1 $R/pp.txt) | $(tail -1 $R/largo.txt) | $(grep TOTAL $R/tar.txt) | $(grep -h '/6\|aciertos' $R/pocas.txt $R/suite.txt | tr '\n' ' ') | $(grep -o 'logp medio.*' $R/sondeos.txt)"
log "fin"
