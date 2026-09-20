#!/bin/bash
# Corre los tests estandar de club-3090 contra NUESTROS contenedores.
#
# Uso:  club_bench.sh <compose.yml> <nombre-contenedor> <ip> [script] [args...]
#   script: bench (default) | verify | verify-full | verify-stress | concurrency-probe | soak-test
#
# Ejemplos:
#   club_bench.sh compose/exp-dflash2-k8.yml genesis-27b-dflash2-k8 172.20.0.241
#   club_bench.sh compose/docker-compose.qwen38-27b-noon-w4a8-v029.yml genesis-27b-v029 172.20.0.228 verify-full
#
# Por que hace falta un puente y no se llaman directo
# ---------------------------------------------------
# Los scripts de club-3090 hablan con el endpoint por `URL` y sacan el nombre del modelo de
# `/v1/models`, asi que apuntarlos es trivial. Lo que NO mandan es la `Authorization`: sus
# composes sirven sin api-key y los nuestros no.
#
# La salida no es sacarle la clave al servidor (eso cambia lo que se mide y deja un server
# abierto en la red) ni editar los scripts de ellos (se pierde en el proximo pull). curl lee
# un `.curlrc` de `$CURL_HOME`, asi que alcanza con apuntar CURL_HOME a un directorio
# temporal con la cabecera adentro: los scripts quedan intactos y la clave nunca aparece en
# una linea de comando (ni en `ps`, ni en el historial).
#
# El nombre del contenedor va aparte porque `bench.sh` scrapea `docker logs` para las
# metricas de spec-decode, y por defecto busca el contenedor de ellos.
#
# OJO con el metodo, que ellos documentan y vale igual aca: nunca compares un arranque FRIO
# contra uno caliente. Reportan ~4% de diferencia de decode en 2x3090 entre una placa fria y
# una en regimen (~1 hora), que es mas que casi cualquier delta que valga la pena reportar.
# El prefill es mucho mas robusto (CV <=0,1% dentro de un arranque).
set -u

CLUB=${CLUB:-/home/usuario/Proyectos/club-3090}
RAIZ=/home/usuario/Proyectos/genesis-vllm-patches

YML=$1; NOMBRE=$2; IP=$3; shift 3
SCRIPT=${1:-bench}; [ $# -gt 0 ] && shift

[ -f "$CLUB/scripts/$SCRIPT.sh" ] || { echo "no existe $CLUB/scripts/$SCRIPT.sh"; exit 1; }

# La clave: del .env del compose, nunca de la linea de comando.
CLAVE=$(grep -E "^VLLM_API_KEY=" "$RAIZ/compose/.env" 2>/dev/null | cut -d= -f2-)
CURLDIR=$(mktemp -d)
trap 'rm -rf "$CURLDIR"' EXIT
if [ -n "$CLAVE" ]; then
  printf 'header = "Authorization: Bearer %s"\n' "$CLAVE" > "$CURLDIR/.curlrc"
  chmod 600 "$CURLDIR/.curlrc"
  # 2026-09-20: el .curlrc NO alcanza. bench.sh solo usa curl para /v1/models; las requests
  # que se miden salen por urllib con headers fijos (solo Content-Type) y daban 401 las 16.
  # Mismo criterio que arriba (no tocar sus scripts, no abrir el server, la clave fuera de
  # la linea de comando): un sitecustomize en el mismo directorio temporal le agrega la
  # Authorization a urllib, SOLO para requests que van al servidor bajo prueba.
  umask 077
  printf '%s' "$CLAVE" > "$CURLDIR/clave"
  cat > "$CURLDIR/sitecustomize.py" <<'PYEOF'
import os, urllib.request
_d = os.path.dirname(os.path.abspath(__file__))
_pref = os.environ.get("GENESIS_BENCH_URL", "")
try:
    _k = open(os.path.join(_d, "clave")).read().strip()
except OSError:
    _k = ""
if _k and _pref:
    _orig = urllib.request.OpenerDirector.open
    def _open(self, fullurl, *a, **kw):
        req = fullurl if isinstance(fullurl, urllib.request.Request) else urllib.request.Request(fullurl)
        if req.full_url.startswith(_pref) and not req.has_header("Authorization"):
            req.add_header("Authorization", "Bearer " + _k)
        return _orig(self, req, *a, **kw)
    urllib.request.OpenerDirector.open = _open
PYEOF
  export PYTHONPATH="$CURLDIR${PYTHONPATH:+:$PYTHONPATH}"
  export GENESIS_BENCH_URL="http://$IP:8320"
fi
export CURL_HOME="$CURLDIR"

# Levantar solo si no esta ya sano: rearrancar para medir tira a la basura el calentamiento.
estado=$(docker inspect "$NOMBRE" --format '{{.State.Health.Status}}' 2>/dev/null)
if [ "$estado" != "healthy" ]; then
  echo "[club_bench] levantando $NOMBRE desde $YML"
  ( cd "$RAIZ/compose" && docker compose -f "$(basename "$YML")" up -d >/dev/null 2>&1 )
  for _ in $(seq 1 75); do
    estado=$(docker inspect "$NOMBRE" --format '{{.State.Health.Status}}' 2>/dev/null)
    [ "$estado" = "healthy" ] && break
    docker logs "$NOMBRE" 2>&1 | grep -q "Engine core initialization failed" && { estado=crash; break; }
    sleep 20
  done
fi
[ "$estado" = "healthy" ] || { echo "[club_bench] $NOMBRE no esta sano ($estado); no mido"; exit 1; }

MODELO=$(curl -s -m 15 "http://$IP:8320/v1/models" | python3 -c \
  'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
[ -n "$MODELO" ] || { echo "[club_bench] no pude leer /v1/models (la clave no entro?)"; exit 1; }

echo "[club_bench] $SCRIPT.sh  ->  $IP:8320  modelo=$MODELO  contenedor=$NOMBRE"
cd "$CLUB" || exit 1
URL="http://$IP:8320" MODEL="$MODELO" CONTAINER="$NOMBRE" \
  bash "scripts/$SCRIPT.sh" "$@"
