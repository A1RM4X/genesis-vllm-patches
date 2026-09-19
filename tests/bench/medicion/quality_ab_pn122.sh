#!/bin/bash
# quality-test.sh de club-3090 (--quick: ToolCall-15 + InstructFollow-15) con y sin PN122.
#
# Por que este y no los oraculos anteriores: ninguno de los dos que use antes discrimina.
# La sombra compara el estado de upstream tras acc-1 tokens contra el de PN122 tras el token
# 0, que solo coinciden si acc==1. Y el A/B de texto greedy es demasiado estricto: 1,9e-4 —el
# error que el propio parche declara— alcanza para dar vuelta un argmax sobre 248.320 tokens.
# Lo unico que separa "redondeo" de "degradacion" es medir si el modelo sigue haciendo bien
# su trabajo.
#
# Los dos brazos corren SIN api-key (los packs piden por su propio cliente, que no manda
# Authorization) y sin puerto publicado: solo por la red interna de docker.
set -u
RAIZ=/home/usuario/Proyectos/genesis-vllm-patches
CLUB=/home/usuario/Proyectos/club-3090
TMP=/home/usuario/.claude/jobs/fc6ace03/tmp
export PATH="$TMP/venv-bl/bin:$PATH"

esperar() {
  until [ "$(docker inspect "$1" --format '{{.State.Health.Status}}' 2>/dev/null)" = "healthy" ]; do
    if docker logs "$1" 2>&1 | grep -q "Engine core initialization failed"; then echo crash; return 1; fi
    sleep 20
  done
  echo healthy
}

brazo() {  # $1=etiqueta  $2=yml  $3=contenedor  $4=ip
  echo "=================== $1 ==================="
  cd "$RAIZ/compose" || return 1
  docker ps -aq --filter "name=genesis-27b-" --format '{{.Names}}' | grep -v "^$3$" \
    | while read n; do docker rm -f "$n" >/dev/null 2>&1; done
  find /dev/shm -maxdepth 1 -type f \( -name "*vllm*" -o -name "*offload*" -o -name "psm*" \
    -o -name "pn122*" \) -delete 2>/dev/null
  docker compose -f "$2" up -d >/dev/null 2>&1
  echo "  salud: $(esperar "$3")"
  [ "$(docker inspect "$3" --format '{{.State.Health.Status}}' 2>/dev/null)" = "healthy" ] || return 1
  docker logs "$3" 2>&1 | grep -E "GPU KV cache size" | grep -v genesis | head -1 | sed 's/^/  /'
  cd "$CLUB" || return 1
  URL="http://$4:8320" MODEL=qwen3.8 CONTAINER="$3" \
    bash scripts/quality-test.sh --quick 2>&1 | tail -30
}

brazo "A — SIN PN122" exp-qa-sin.yml genesis-27b-qa-sin 172.20.0.153
echo
brazo "B — CON PN122" exp-qa-con.yml genesis-27b-qa-con 172.20.0.154
