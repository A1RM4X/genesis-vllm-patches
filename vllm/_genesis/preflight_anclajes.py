# SPDX-License-Identifier: Apache-2.0
"""Pre-vuelo de migracion: verifica los anclajes de texto contra otra version de vLLM.

Cada parche de cirugia de texto busca un ``anchor`` exacto dentro de un archivo de
vLLM. Cuando upstream mueve una linea el ancla no aparece y ``TextPatcher`` devuelve
SKIPPED: el servidor arranca igual, pero sin el parche y **sin gritar**. Eso es lo
caro de una migracion — no lo que revienta, sino lo que se apaga en silencio.

Esto contesta la pregunta antes de bajar nada: junta las anclas de todos los
``wiring/**/patch_*.py`` sin aplicar ninguna, y las cuenta contra un arbol de
fuentes de la version destino.

Uso::

    # bajar el arbol de referencia (solo los archivos que tocamos)
    python -m vllm._genesis.preflight_anclajes --bajar v0.29.0 --arbol /tmp/v029

    # y verificar
    python -m vllm._genesis.preflight_anclajes --arbol /tmp/v029

Salida: una linea por sub-parche. ``ROTO`` es un ancla con 0 ocurrencias (el parche
se va a apagar solo) y ``AMBIGUO`` una con mas de una (reemplaza la primera, que
puede no ser la que queremos).
"""

from __future__ import annotations

import argparse
import importlib
import os
import pathlib
import sys
import urllib.request

RAW = "https://raw.githubusercontent.com/vllm-project/vllm/{tag}/vllm/{rel}"


def _wiring_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent / "wiring"


def _modulos() -> list[str]:
    base = _wiring_dir()
    mods = []
    for f in sorted(base.rglob("patch_*.py")):
        rel = f.relative_to(base.parent.parent.parent)  # desde la raiz del paquete vllm
        mods.append(str(rel.with_suffix("")).replace(os.sep, "."))
    return mods


def recolectar() -> list[tuple[str, str, list[tuple[str, str, bool]], list[str]]]:
    """Importa cada wiring y le saca las anclas SIN aplicar nada.

    Devuelve ``[(parche, destino, [(sub, ancla, requerida), ...], derivas), ...]``.
    """
    from vllm._genesis.wiring import text_patch as tp

    recogido: list[tuple[str, str, list[tuple[str, str, bool]], list[str]]] = []

    real_apply = tp.TextPatcher.apply

    def espia(self):
        recogido.append((
            self.patch_name, self.target_file,
            [(s.name, s.anchor, s.required) for s in self.sub_patches],
            list(self.upstream_drift_markers),
        ))
        return tp.TextPatchResult.APPLIED, None

    # Las transacciones multi-archivo llaman apply() de cada patcher por dentro, asi que
    # con espiar TextPatcher.apply alcanza para las dos formas.
    tp.TextPatcher.apply = espia

    # El dispatcher decide por variables de entorno; aca queremos ver TODAS las anclas,
    # incluso las de parches que hoy estan apagados.
    from vllm._genesis import dispatcher
    real_should = dispatcher.should_apply
    dispatcher.should_apply = lambda *a, **k: (True, "preflight: forzado")

    try:
        for m in _modulos():
            try:
                mod = importlib.import_module(m)
            except Exception as e:                       # noqa: BLE001
                print(f"  (no importa {m}: {type(e).__name__}: {e})", file=sys.stderr)
                continue
            fn = getattr(mod, "apply", None)
            if fn is None:
                continue
            try:
                fn()
            except Exception as e:                       # noqa: BLE001
                print(f"  (apply() de {m} tiro {type(e).__name__}: {e})", file=sys.stderr)
    finally:
        tp.TextPatcher.apply = real_apply
        dispatcher.should_apply = real_should
    return recogido


def _rel_vllm(target: str) -> str:
    """De ``/usr/.../site-packages/vllm/v1/x.py`` a ``v1/x.py``."""
    p = pathlib.Path(target).resolve()
    partes = list(p.parts)
    # el ultimo componente 'vllm' del camino es la raiz del paquete instalado
    for i in range(len(partes) - 1, -1, -1):
        if partes[i] == "vllm":
            return str(pathlib.Path(*partes[i + 1:]))
    return p.name


def bajar(tag: str, arbol: str, rels: list[str]) -> None:
    for rel in sorted(set(rels)):
        dest = pathlib.Path(arbol) / rel
        if dest.is_file():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = RAW.format(tag=tag, rel=rel)
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                dest.write_bytes(r.read())
            print(f"  bajado {rel}")
        except Exception as e:                           # noqa: BLE001
            print(f"  FALTA {rel} en {tag}: {e}")


def _vecindad(src: str, ancla: str, contexto: int = 6) -> str:
    """Region del archivo nuevo mas parecida al ancla que ya no aparece.

    Sin esto, arreglar un ancla rota es buscar a ojo en un archivo de mil lineas.
    Ancla el diff en la linea del ancla que mas se parezca a alguna del destino.
    """
    import difflib

    lineas_a = [l for l in ancla.splitlines() if l.strip()]
    if not lineas_a:
        return "(ancla vacia)"
    destino = src.splitlines()
    mejor_i, mejor_r = None, 0.0
    for la in lineas_a:
        for i, ld in enumerate(destino):
            r = difflib.SequenceMatcher(None, la.strip(), ld.strip()).ratio()
            if r > mejor_r:
                mejor_r, mejor_i = r, i
    if mejor_i is None or mejor_r < 0.5:
        return "(nada parecido en el archivo nuevo — la funcion puede haberse ido)"
    ini = max(0, mejor_i - contexto)
    fin = min(len(destino), mejor_i + contexto + len(lineas_a))
    out = [f"    ~{mejor_r:.0%} de parecido, alrededor de la linea {mejor_i + 1}:"]
    for n in range(ini, fin):
        marca = ">>" if n == mejor_i else "  "
        out.append(f"    {marca} {n + 1:5d} | {destino[n]}")
    return "\n".join(out)


def verificar(arbol: str, explicar: bool = False) -> int:
    datos = recolectar()
    rels = [_rel_vllm(t) for _, t, _, _ in datos]
    raiz = pathlib.Path(arbol)

    rotos = ambiguos = ok = 0
    sin_archivo: list[str] = []
    obsoletos: list[str] = []
    for parche, target, subs, derivas in datos:
        rel = _rel_vllm(target)
        f = raiz / rel
        if not f.is_file():
            sin_archivo.append(f"{parche}  ->  {rel}")
            continue
        src = f.read_text(errors="replace")
        # Mismo orden que TextPatcher: si upstream ya trae el arreglo, el parche se salta
        # con un mensaje claro y sus anclas rotas no son un problema, son la prueba.
        deriva = next((d for d in derivas if d and d in src), None)
        if deriva is not None:
            obsoletos.append(f"{parche}  (upstream ya lo trae: {deriva!r})")
            continue
        malas = []
        for nombre, ancla, requerida in subs:
            # Un ancla puede traer variantes por version de vLLM: cuenta la que matchee.
            variantes = [ancla] if isinstance(ancla, str) else list(ancla)
            presentes = [v for v in variantes if src.count(v) == 1]
            if presentes:
                c = 1
                ancla = presentes[0]
            else:
                ancla = variantes[0]
                c = max((src.count(v) for v in variantes), default=0)
            if c == 1:
                ok += 1
            elif c == 0:
                rotos += 1
                malas.append(f"ROTO     {nombre}{' (requerida)' if requerida else ''}")
                if explicar:
                    malas.append(_vecindad(src, ancla))
            else:
                ambiguos += 1
                malas.append(f"AMBIGUO  {nombre}: {c} ocurrencias")
        if malas:
            print(f"\n{parche}  [{rel}]")
            for m in malas:
                print(f"    {m}")

    if obsoletos:
        print("\nParches que upstream ya absorbio (se saltan solos, no hay nada que arreglar):")
        for o in obsoletos:
            print(f"    {o}")

    if sin_archivo:
        print("\nArchivos que no estan en el arbol de referencia "
              "(o no existen en esa version):")
        for s in sin_archivo:
            print(f"    {s}")

    print(f"\nResumen: {ok} anclas OK, {rotos} rotas, {ambiguos} ambiguas, "
          f"{len(obsoletos)} parches obsoletos, {len(sin_archivo)} archivos ausentes.")
    return 1 if (rotos or ambiguos) else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arbol", required=True, help="directorio con las fuentes de la version destino")
    ap.add_argument("--bajar", metavar="TAG", help="bajar antes de github (ej: v0.29.0)")
    ap.add_argument("--explicar", action="store_true",
                    help="para cada ancla rota, mostrar la region equivalente del archivo nuevo")
    a = ap.parse_args()
    if a.bajar:
        rels = [_rel_vllm(t) for _, t, _, _ in recolectar()]
        print(f"Bajando {len(set(rels))} archivos de {a.bajar} a {a.arbol}")
        bajar(a.bajar, a.arbol, rels)
    return verificar(a.arbol, explicar=a.explicar)


if __name__ == "__main__":
    raise SystemExit(main())
