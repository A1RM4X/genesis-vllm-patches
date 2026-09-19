#!/bin/bash
# N corridas de prod_largos, REINICIANDO entre cada una, contando cuantas revientan.
# Reiniciar es obligatorio: el EngineCore muere con el primer IndexError, asi que sin reinicio
# el contador satura en 1 y las corridas siguientes no miden nada. Ese fue el error que
# invalido la primera biseccion.
cd /home/usuario/Proyectos/genesis-vllm-patches/compose
YML=$1; NOMBRE=$2; IP=$3; N=${4:-3}; K=${VLLM_API_KEY:?falta VLLM_API_KEY}
crash=0; limpio=0
for n in $(seq 1 $N); do
  docker compose -f $YML down >/dev/null 2>&1
  find /dev/shm -maxdepth 1 -type f \( -name "*vllm*" -o -name "*offload*" -o -name "psm*" \) -delete 2>/dev/null
  docker compose -f $YML up -d >/dev/null 2>&1
  ok=no
  for i in $(seq 1 50); do curl -sf -H "Authorization: Bearer $K" -m 5 "http://$IP/health" >/dev/null 2>&1 && { ok=si; break; }; sleep 20; done
  [ "$ok" = no ] && { echo "  $n: NO LEVANTO"; continue; }
  (cd ../tests/bench/medicion && VLLM_HOST=http://$IP timeout 1200 python3 prod_largos.py >/dev/null 2>&1)
  c=$(docker logs $NOMBRE 2>&1 | grep -c "IndexError: list index out of range")
  if [ "$c" -gt 0 ]; then crash=$((crash+1)); echo "  $n: CRASH"; else limpio=$((limpio+1)); echo "  $n: limpio"; fi
done
echo "== $NOMBRE: $crash crash, $limpio limpios de $N =="
