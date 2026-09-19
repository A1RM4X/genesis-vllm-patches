#!/usr/bin/env bash
# Publica en origin la historia reescrita que purgo la credencial (2026-09-19).
#
# Por defecto hace DRY-RUN: muestra exactamente que va a pasar y no toca nada.
# Para ejecutarlo de verdad:  bash scripts/publicar-historia-reescrita.sh --ejecutar
#
# QUE HACE, Y POR QUE
# -------------------
# `git filter-repo` reescribio TODAS las ramas locales, asi que todos los hashes cambiaron.
# Un push normal seria rechazado (no es fast-forward); hay que forzar. Eso es intencional:
# el objetivo es justamente reemplazar la historia que tiene la clave adentro.
#
# Las ramas de dependabot se BORRAN en vez de pisarse. Dos razones:
#   * sus PRs quedan rotos igual despues de una reescritura, asi que pisarlas no salva nada;
#   * una de ellas (python-minor-patch-61f63fc237) no existe en local, asi que si no se borra
#     se queda con la historia VIEJA y la credencial sigue visible ahi.
# Dependabot las vuelve a crear solo contra la historia nueva.
#
# LO QUE ESTO ROMPE, dicho antes de correrlo
#   * cualquier clon existente queda desincronizado y hay que re-clonar (no basta con pull)
#   * los PRs abiertos contra las ramas borradas se cierran
#   * los hashes de TODOS los commits cambian: links a commits viejos dejan de resolver
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
EJECUTAR=0
[[ "${1:-}" == "--ejecutar" ]] && EJECUTAR=1

PISAR=(main opt/super-kernels-int32 update/v0.27.1)
BORRAR=(
  dependabot/github_actions/actions/checkout-7
  dependabot/github_actions/actions/github-script-9
  dependabot/github_actions/actions/upload-artifact-7
  dependabot/pip/hypothesis-gte-6.165.10
  dependabot/pip/pytest-gte-9.1.1
  dependabot/pip/python-minor-patch-61f63fc237
)

echo "=============================================================="
echo " Publicar historia reescrita  $([[ $EJECUTAR == 1 ]] && echo '*** EJECUTANDO ***' || echo '(DRY-RUN)')"
echo "=============================================================="

# --- Guardas: si alguna falla, no se toca el remoto ---
echo
echo "[1/4] Verificando que la credencial no este en NINGUN objeto local..."
# El patron se arma por partes A PROPOSITO: si este script llevara el literal, el propio
# script se volveria un blob "sucio" y la guarda de abajo se dispararia sola. Paso.
PATRON="super-secret""-key-123"
sucios=$(git rev-list --all --objects 2>/dev/null | awk '{print $1}' | sort -u \
  | git cat-file --batch-check='%(objectname) %(objecttype)' 2>/dev/null \
  | awk '$2=="blob"{print $1}' | git cat-file --batch 2>/dev/null \
  | grep -ac "$PATRON")
if [[ "$sucios" != "0" ]]; then
  echo "  ABORTA: quedan $sucios blobs con la credencial. La reescritura no esta completa."
  exit 1
fi
echo "  ok: 0 blobs con la credencial"

echo "[2/4] Verificando que exista el backup..."
BK=$(ls -d /home/usuario/Proyectos/backup-genesis-* 2>/dev/null | tail -1)
if [[ -z "$BK" || ! -f "$BK/repo-completo.bundle" ]]; then
  echo "  ABORTA: no encuentro el bundle de backup. Crealo antes:"
  echo "    git bundle create /ruta/backup.bundle --all"
  exit 1
fi
echo "  ok: $BK"

echo "[3/4] Verificando que el arbol de trabajo no tenga cambios sin commitear..."
pendientes=$(git status --porcelain | grep -vE "assets/vllm" | wc -l)
if [[ "$pendientes" != "0" ]]; then
  echo "  ABORTA: hay $pendientes cambios sin commitear. Commitealos o descartalos primero."
  git status --short | grep -vE "assets/vllm" | head -5
  exit 1
fi
echo "  ok: arbol limpio (el submodulo assets/vllm se ignora a proposito)"

echo "[4/4] Plan:"
for b in "${PISAR[@]}"; do
  L=$(git rev-parse --short "$b" 2>/dev/null || echo "NO-EXISTE")
  R=$(git ls-remote --heads origin "$b" 2>/dev/null | cut -c1-8)
  printf "  PISAR   %-30s  %s -> %s\n" "$b" "${R:-ausente}" "$L"
done
for b in "${BORRAR[@]}"; do
  R=$(git ls-remote --heads origin "$b" 2>/dev/null | cut -c1-8)
  [[ -n "$R" ]] && printf "  BORRAR  %-30s  %s\n" "$b" "$R"
done

if [[ $EJECUTAR == 0 ]]; then
  echo
  echo "DRY-RUN: no se toco nada."
  echo "Para ejecutarlo:  bash scripts/publicar-historia-reescrita.sh --ejecutar"
  exit 0
fi

echo
echo "--- pisando ramas ---"
for b in "${PISAR[@]}"; do
  git rev-parse --verify -q "$b" >/dev/null || { echo "  (salteo $b: no existe local)"; continue; }
  echo "  push --force origin $b"
  git push --force origin "$b:$b" || { echo "  FALLO en $b — revisa y volve a correr"; exit 1; }
done

echo "--- borrando ramas de dependabot ---"
for b in "${BORRAR[@]}"; do
  git ls-remote --heads origin "$b" 2>/dev/null | grep -q . || continue
  echo "  delete origin $b"
  git push origin --delete "$b" || echo "  (no se pudo borrar $b, seguimos)"
done

echo
echo "--- verificacion final contra el remoto ---"
git ls-remote --heads origin | awk '{print "  " substr($1,1,8), $2}'
echo
echo "LISTO. Lo que queda por hacer a mano:"
echo "  1. En GitHub, Settings -> Danger Zone -> cambiar visibilidad a Public."
echo "  2. GitHub puede conservar objetos viejos alcanzables desde PRs cerrados."
echo "     La credencial YA ESTA ROTADA, asi que el riesgo real es nulo, pero si"
echo "     querés que desaparezcan del todo hay que pedirle un GC a GitHub Support."
echo "  3. Avisar a cualquiera con un clon: tiene que RE-CLONAR, no alcanza con pull."
