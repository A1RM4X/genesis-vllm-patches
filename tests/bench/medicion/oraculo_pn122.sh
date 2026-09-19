#!/bin/bash
# Corre el oraculo en sombra de PN122: el bit 32 de /dev/shm/pn122_sync hace que las capas
# 0, 1, 2 y 44 evolucionen TAMBIEN con el kernel de upstream, y reporta
#   err_salida = ||o_ref - o|| / ||o_ref||
# La referencia del propio parche, validada aislada, es 1,9e-4 (redondeo fp16).
set -u
C=genesis-27b-pn122-eager
IP=172.20.0.151:8320
K=<REDACTADO: clave rotada 2026-09-19>

echo "== esperando a que levante =="
until [ "$(docker inspect $C --format '{{.State.Health.Status}}' 2>/dev/null)" = "healthy" ] \
   || docker logs $C 2>&1 | grep -q "Engine core initialization failed"; do sleep 20; done
est=$(docker inspect $C --format '{{.State.Health.Status}}' 2>/dev/null)
echo "salud: $est"
[ "$est" = "healthy" ] || { docker logs $C 2>&1 | grep -iE "error|assert" | grep -viE "genesis|\|" | tail -5; exit 1; }

echo; echo "== KV con PN122 (num_speculative_blocks=0) =="
docker logs $C 2>&1 | grep -E "DIAG KV" | sed 's/.*\[DIAG KV\] //' | awk '!seen[$0]++' \
  | grep -E "grp 0 |grp .*SlidingW|TOTAL .*bloques/req" | head -4
docker logs $C 2>&1 | grep -E "GPU KV cache size" | grep -v genesis | head -1

echo; echo "== prendo el oraculo (bit 32) =="
docker exec $C sh -c 'echo 32 > /dev/shm/pn122_sync; cat /dev/shm/pn122_sync'

echo; echo "== genero trafico para que corra el kernel spec =="
for i in 1 2 3; do
  curl -s -m 300 http://$IP/v1/completions -H 'Content-Type: application/json' \
    -H "Authorization: Bearer $K" \
    -d "{\"model\":\"qwen3.8\",\"prompt\":\"Corrida $i. Enumera veinte ciudades de Europa, una por linea, con su pais y un dato historico.\",\"max_tokens\":250,\"temperature\":0}" \
    > /home/usuario/.claude/jobs/fc6ace03/tmp/pn122_salida_$i.json
done
echo "  salida de la corrida 1:"
python3 -c "
import json
d=json.load(open('/home/usuario/.claude/jobs/fc6ace03/tmp/pn122_salida_1.json'))
print('   ', repr(d['choices'][0]['text'][:150]))"

echo; echo "== err_salida reportado por la sombra =="
docker logs $C 2>&1 | grep "PN122 sombra" | tail -20 | sed 's/.*\[PN122 sombra\] /  /'
echo
echo "== resumen del error =="
docker logs $C 2>&1 | grep -oE "err_salida=[0-9.e+-]+" | sed 's/err_salida=//' | python3 -c "
import sys
v=[float(x) for x in sys.stdin if x.strip()]
if not v:
    print('  la sombra no reporto nada — el bit no engancho o no corrio el kernel spec')
else:
    v.sort()
    print(f'  {len(v)} muestras | min {v[0]:.2e} | mediana {v[len(v)//2]:.2e} | max {v[-1]:.2e}')
    print(f'  referencia del parche (validacion aislada, redondeo fp16): 1.9e-04')
    print('  VEREDICTO:', 'dentro de la referencia' if v[-1] < 1e-3 else 'FUERA: la cinta diverge del kernel de upstream')
"
