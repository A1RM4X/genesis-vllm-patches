#!/bin/bash
R=/home/usuario/Proyectos/genesis-vllm-patches/tests/bench/medicion/resultados_noche
RT=/home/usuario/Proyectos/genesis-vllm-patches/tests/bench/medicion/resultados_tres
CD=/home/usuario/Proyectos/genesis-vllm-patches/compose
G=/home/usuario/Proyectos/genesis-vllm-patches/vllm/_genesis
CN=genesis-27b-qwen38-ar4ikov-awq
A="Authorization: Bearer ${VLLM_API_KEY:?falta VLLM_API_KEY}"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $R/estado.log; }
until grep -q "^\[.*\] fin" $RT/estado.log 2>/dev/null; do sleep 60; done
bajar(){ for f in $CD/docker-compose.*.yml; do grep -q "container_name: $CN" $f && docker compose -f $f down >/dev/null 2>&1; done; docker rm -f $CN >/dev/null 2>&1; }
arrancar(){ bajar
  docker run --rm -v /home/usuario/.cache/vllm:/c alpine rm -rf /c/qwen38-27b-noon-w4a8/torch_compile_cache
  docker compose -f $1 up -d >/dev/null 2>&1; sleep 60
  for i in $(seq 1 200); do curl -s -m 5 -H "$A" 127.0.0.1:8320/v1/models | grep -q '"id"' && return 0; docker ps | grep -q $CN || return 1; sleep 10; done; return 1; }
medir(){ T=$1; O=$R/$T; mkdir -p $O; docker logs $CN > $O/arranque.log 2>&1
  cd /tmp/g115
  timeout 900 python3 sanidad.py > $O/sanidad.txt 2>&1
  timeout 1800 python3 pp.py $T > $O/pp.txt 2>&1
  timeout 1800 python3 largo.py $T > $O/largo.txt 2>&1
  timeout 1800 python3 tar_multi.py $T > $O/tar.txt 2>&1
  timeout 1800 python3 pocas.py $T > $O/pocas.txt 2>&1
  timeout 3600 python3 suite_calidad.py > $O/suite.txt 2>&1
  timeout 3600 python3 sondeos.py $T 160 > $O/sondeos.txt 2>&1
  log "$T: $(tail -1 $O/pp.txt) | $(tail -1 $O/largo.txt) | $(grep TOTAL $O/tar.txt) | $(grep -h '/6\|aciertos' $O/pocas.txt $O/suite.txt | tr '\n' ' ') | $(grep -o 'logp medio.*' $O/sondeos.txt) | KV $(grep -o 'GPU KV cache size: [0-9,]*' $O/arranque.log | tail -1)"; }

log "inicio noche"
# 1) KV int4 con y sin rotacion (Triton + PN124)
for v in sin had wush; do
  if arrancar $CD/docker-compose.rot-i4-$v.yml; then medir noche_i4_$v; else log "i4_$v NO ARRANCO"; docker logs --tail 150 $CN > $R/i4_${v}_fallo.log 2>&1; fi
done
# 2) piso de ruido: dos repeticiones mas de la base (KV fp8, sin rotar)
if arrancar $CD/docker-compose.qwen38-27b-noon-w4a8.yml; then
  cd /tmp/g115; timeout 900 python3 sanidad.py > /dev/null 2>&1
  for r in 1 2; do timeout 3600 python3 sondeos.py noche_ruido$r 160 > $R/ruido$r.txt 2>&1; done
  log "ruido: $(grep -ho 'logp medio.*' $R/ruido1.txt $R/ruido2.txt | tr '\n' ' ')"
fi
cd /tmp/g115
python3 sondeos.py --comparar noon_w4a8_pn125 rot_fp8_control noche_ruido1 noche_ruido2 > $R/ruido_comparacion.txt 2>&1
python3 sondeos.py --comparar noon_w4a8_pn125 noche_i4_sin noche_i4_had noche_i4_wush noche_ruido1 > $R/i4_comparacion.txt 2>&1
# 3) kernels con GPU libre, forma real
bajar
for Mv in 7488 2048; do
  docker run --rm --gpus '"device=0"' -v $G:/usr/local/lib/python3.12/dist-packages/vllm/_genesis:ro -v /home/usuario/Proyectos/genesis-vllm-patches/tests/proto:/p:ro -e HOME=/tmp --entrypoint python3 vllm/vllm-openai:v0.27.1 /p/sk_e2e_bench.py $Mv 2>&1 | grep "M=" >> $R/kernels.txt
done
docker run --rm --gpus '"device=0"' -v $G:/usr/local/lib/python3.12/dist-packages/vllm/_genesis:ro -e HOME=/tmp --entrypoint python3 vllm/vllm-openai:v0.27.1 -m vllm._genesis.kernels.sk16_gemm_f16 --bench 2>&1 | grep "ms" >> $R/kernels.txt
log "kernels: $(cat $R/kernels.txt | tr '\n' ';')"
# 4) dejar noon W4A8 + PN125 arriba
docker run --rm -v /home/usuario/.cache/vllm:/c alpine rm -rf /c/qwen38-27b-noon-w4a8/torch_compile_cache
docker compose -f $CD/docker-compose.qwen38-27b-noon-w4a8.yml up -d >/dev/null 2>&1
log "fin noche (queda arriba noon W4A8 + PN125, KV fp8)"
