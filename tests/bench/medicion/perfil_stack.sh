#!/usr/bin/env bash
# perfil_stack.sh <label> <compose> [env extra...]: traza torch de decode a 57k (prefijo cacheado)
# y de un prefill frio de ~16k. Deja las trazas en tests/bench/medicion/trazas/<label>_{decode,prefill}.
set -uo pipefail
LABEL=$1; COMPOSE=$2; shift 2
cd /home/usuario/Proyectos/genesis-vllm-patches/compose || exit 1
MED=/home/usuario/Proyectos/genesis-vllm-patches/tests/bench/medicion
H='-H Content-Type:application/json -H Authorization:Bearer_x'
esperar() { for i in $(seq 1 70); do L=$(docker logs --tail 200 genesis-27b-qwen38-ar4ikov-awq 2>&1); echo "$L" | grep -q "Application startup complete" && return 0; echo "$L" | grep -qE "Traceback|EngineCore failed" && { echo FALLA; return 1; }; sleep 20; done; return 1; }
req() { docker exec genesis-27b-qwen38-ar4ikov-awq python3 -c "$1"; }
for FASE in decode prefill; do
  docker compose -f $COMPOSE down >/dev/null 2>&1
  if [ $FASE = decode ]; then DEL=6; MAX=60; else DEL=0; MAX=6; fi
  docker ps --format "{{.Names}}" | grep -q genesis-27b && { echo "no se bajo el server"; exit 1; }
  env "$@" PROF_KIND='"torch"' PROF_DIR=/traces/${LABEL}_$FASE PROF_DELAY=$DEL PROF_MAX=$MAX docker compose -f $COMPOSE up -d >/dev/null 2>&1
  esperar || exit 1
  echo "[$LABEL/$FASE] listo"
  req "
import json,urllib.request,glob,time
H={'Content-Type':'application/json','Authorization':'Bearer <REDACTADO: clave rotada 2026-09-19>'}
def chat(txt,mt):
    b=json.dumps({'model':'qwen3.8','messages':[{'role':'user','content':txt}],'max_tokens':mt,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}).encode()
    return json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8320/v1/chat/completions',b,H),timeout=900))
fs=sorted(glob.glob('/usr/local/lib/python3.12/dist-packages/vllm/_genesis/**/*.py',recursive=True))
txt=''.join(open(f,errors='ignore').read() for f in fs)
largo=txt[:200000]+'\n\nResumí en detalle que hace este codigo, modulo por modulo.'
corto=txt[300000:356000]+'\n\nDecí solo OK.'
fase='$FASE'
if fase=='decode':
    chat(largo,8)
    urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8320/start_profile',b'',H,method='POST'),timeout=60)
    r=chat(largo,300); print('decode tokens', r['usage'])
else:
    chat('Explicame paginacion en detalle.',32)
    urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8320/start_profile',b'',H,method='POST'),timeout=60)
    t=time.time(); r=chat(corto,1); print('prefill', r['usage'], round(time.time()-t,2),'s')
urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8320/stop_profile',b'',H,method='POST'),timeout=600)
"
  for i in $(seq 1 40); do n=$(ls $MED/trazas/${LABEL}_$FASE 2>/dev/null | grep -c json); [ "$n" -gt 0 ] && break; sleep 10; done
  sleep 20; ls -la $MED/trazas/${LABEL}_$FASE | tail -3
done
docker compose -f $COMPOSE down >/dev/null 2>&1
