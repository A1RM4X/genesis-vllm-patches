#!/bin/bash
# A/B INTERCALADO de decode para PN122: A/B/A/B, con 3 calentamientos y 5 corridas medidas
# por arranque.
#
# Por que intercalado y no A-luego-B: club-3090 documenta que una pareja secuencial es "the
# one design that cannot distinguish the patch from the position" — reportan ~4% de deriva de
# decode en 2x3090 entre una placa fria y una en regimen (~1 hora). Alternando A/B/A/B, la
# diferencia entre A1 y A2 mide esa deriva y dice si el delta A-vs-B se le puede creer.
#
# Prompt FIJO y temperature=0: con max_tokens fijo los dos brazos generan la misma cantidad de
# tokens, asi que decode_tps = completion / (wall - ttft) compara trabajo igual. El
# prefix-cache pega en el TTFT, no en el decode, y pega igual en los dos brazos.
set -u
RAIZ=/home/usuario/Proyectos/genesis-vllm-patches
TMP=/home/usuario/.claude/jobs/fc6ace03/tmp
K=${VLLM_API_KEY:?falta VLLM_API_KEY}
SALIDA=400
CAL=3        # calentamientos
MED=5        # corridas medidas

cat > $TMP/_una.py <<'PY'
import json, sys, time, urllib.request
ip, salida = sys.argv[1], int(sys.argv[2])
H = {'Content-Type': 'application/json', 'Authorization': 'Bearer ${VLLM_API_KEY:?falta VLLM_API_KEY}'}
P = ("Explica en detalle como funciona un cache de varios niveles en una CPU moderna: "
     "politicas de reemplazo, coherencia entre nucleos, prefetch, y el impacto de la "
     "localidad espacial y temporal en el rendimiento de un programa real.")
c = json.dumps({"model": "qwen3.8", "prompt": P, "max_tokens": salida,
                "temperature": 0, "stream": True}).encode()
t0 = time.time(); ttft = None; ultimo = t0; n = 0
for linea in urllib.request.urlopen(
        urllib.request.Request(f'http://{ip}/v1/completions', c, H), timeout=600):
    if linea.startswith(b"data: ") and b"[DONE]" not in linea:
        if ttft is None: ttft = time.time() - t0
        n += 1; ultimo = time.time()
dec = ultimo - t0 - (ttft or 0)
print(f"{salida/dec:.2f} {ttft:.3f}" if dec > 0 else "0 0")
PY

esperar() {
  until [ "$(docker inspect "$1" --format '{{.State.Health.Status}}' 2>/dev/null)" = "healthy" ]; do
    docker logs "$1" 2>&1 | grep -q "Engine core initialization failed" && return 1
    sleep 20
  done
}

corrida() {  # $1=etiqueta  $2=yml  $3=contenedor  $4=ip
  cd "$RAIZ/compose" || return 1
  docker ps -aq --filter "name=genesis-27b-ab" --format '{{.Names}}' \
    | while read n; do docker rm -f "$n" >/dev/null 2>&1; done
  find /dev/shm -maxdepth 1 -type f \( -name "*vllm*" -o -name "*offload*" -o -name "psm*" \
    -o -name "pn122*" \) -delete 2>/dev/null
  docker compose -f "$2" up -d >/dev/null 2>&1
  esperar "$3" || { echo "$1 NO LEVANTO"; return 1; }
  for _ in $(seq 1 $CAL); do python3 $TMP/_una.py "$4:8320" $SALIDA >/dev/null 2>&1; done
  vals=""
  for _ in $(seq 1 $MED); do
    vals="$vals $(python3 $TMP/_una.py "$4:8320" $SALIDA 2>/dev/null | cut -d' ' -f1)"
  done
  kv=$(docker logs "$3" 2>&1 | grep -oE "GPU KV cache size: [0-9,]+" | head -1 | grep -oE "[0-9,]+")
  echo "$1|$kv|$vals"
}

echo "== A/B intercalado: A sin PN122, B con PN122 =="
echo "   $CAL calentamientos + $MED medidas por arranque, prompt fijo, greedy, $SALIDA tokens"
echo
{
corrida "A1" exp-ab-sin.yml genesis-27b-ab-sin 172.20.0.155
corrida "B1" exp-ab-con.yml genesis-27b-ab-con 172.20.0.156
corrida "A2" exp-ab-sin.yml genesis-27b-ab-sin 172.20.0.155
corrida "B2" exp-ab-con.yml genesis-27b-ab-con 172.20.0.156
} | tee $TMP/ab_decode_crudo.txt

echo
python3 - $TMP/ab_decode_crudo.txt <<'PY'
import statistics as st, sys
filas = {}
for l in open(sys.argv[1]):
    if "|" not in l: continue
    et, kv, vals = l.strip().split("|")
    v = [float(x) for x in vals.split() if x]
    if v: filas[et] = (kv, v)
print("  arranque   KV tokens    n   media     std    CV      valores")
for et in ("A1", "B1", "A2", "B2"):
    if et not in filas: continue
    kv, v = filas[et]
    m, s = st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)
    print(f"  {et:9s} {kv:>11s} {len(v):3d}  {m:7.1f} {s:6.2f}  {100*s/m:4.1f}%  "
          + " ".join(f"{x:.0f}" for x in v))
A = filas.get("A1", ("", []))[1] + filas.get("A2", ("", []))[1]
B = filas.get("B1", ("", []))[1] + filas.get("B2", ("", []))[1]
if A and B:
    ma, mb = st.mean(A), st.mean(B)
    print(f"\n  A (sin PN122): {ma:.1f} tok/s   B (con PN122): {mb:.1f} tok/s   "
          f"delta {100*(mb-ma)/ma:+.1f}%")
    if "A1" in filas and "A2" in filas:
        a1, a2 = st.mean(filas["A1"][1]), st.mean(filas["A2"][1])
        deriva = 100 * abs(a2 - a1) / a1
        print(f"  deriva entre arranques del MISMO brazo (A1 vs A2): {deriva:.1f}%")
        print("  VEREDICTO:", "el delta A-vs-B supera la deriva: se le puede creer"
              if abs(100*(mb-ma)/ma) > deriva else
              "el delta NO supera la deriva entre arranques: no concluyente")
PY
