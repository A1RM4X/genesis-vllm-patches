#!/usr/bin/env python3
"""Radiografia estructural de un kernel en PTX: bloques basicos, lazos y que hace cada seccion.

Contar instrucciones sueltas no dice donde estan. Esto arma el grafo de bloques basicos, detecta
los lazos por los saltos hacia atras, y clasifica cada bloque por lo que hace (cargar de global,
desempaquetar nibbles, multiplicar en el tensor core, reducir, escribir). Asi se ve si la grasa
esta adentro del lazo caliente — que es lo unico que importa — o en el prologo.

Uso: estructura.py <kernel.ptx>
"""
from __future__ import annotations

import collections
import re
import sys

# Que "hace" una instruccion, para poder etiquetar un bloque por su contenido.
ROLES = [
    ("mma",          r"^mma\.|^wmma\."),
    ("carga global", r"^cp\.async|^ld\.global"),
    ("shared",       r"^ld\.shared|^st\.shared|^ldmatrix"),
    ("barrera",      r"^bar\.|^barrier|^fence|^membar"),
    ("bits",         r"^and\.|^or\.|^xor\.|^shl\.|^shr\.|^lop3|^prmt|^bfe|^bfi|^not\."),
    ("fp",           r"\.f32|\.f16|\.bf16|^cvt\.rn\.f|^rcp\.|^div\.rn"),
    ("div entera",   r"^div\.[su]|^rem\.[su]"),
    ("mul entera",   r"^mul\.|^mad\.|^mul24"),
    ("suma entera",  r"^add\.|^sub\.|^addc|^subc"),
    ("salto",        r"^bra|^ret|^call|^@"),
    ("mov/sel",      r"^mov\.|^selp|^setp|^cvt\."),
    ("otros",        r"."),
]


def rol(op: str) -> str:
    for nombre, pat in ROLES:
        if re.search(pat, op):
            return nombre
    return "otros"


def parsear(ruta: str):
    """-> (bloques, orden) donde bloques[etiqueta] = lista de opcodes."""
    bloques: dict[str, list[str]] = collections.OrderedDict()
    orden: list[str] = []
    actual = "entrada"
    bloques[actual] = []
    orden.append(actual)
    saltos: dict[str, list[str]] = collections.defaultdict(list)
    for linea in open(ruta, errors="ignore"):
        s = linea.strip()
        if not s or s.startswith("//"):
            continue
        m = re.match(r"^(\$L__\w+):", s)
        if m:
            actual = m.group(1)
            if actual not in bloques:
                bloques[actual] = []
                orden.append(actual)
            continue
        if s.startswith((".", "{", "}", ")", "(")) or s.endswith(","):
            continue
        mo = re.match(r"^(?:@!?%p\d+\s+)?([a-z][a-z0-9_.]*)", s)
        if not mo:
            continue
        op = mo.group(1)
        bloques[actual].append(op)
        if op.startswith("bra"):
            d = re.search(r"(\$L__\w+)", s)
            if d:
                saltos[actual].append(d.group(1))
    return bloques, orden, saltos


def main() -> None:
    ruta = sys.argv[1] if len(sys.argv) > 1 else "decode_ptx.txt"
    bloques, orden, saltos = parsear(ruta)
    pos = {b: i for i, b in enumerate(orden)}

    # Un salto a un bloque ANTERIOR cierra un lazo; el destino es la cabeza.
    cabezas: dict[str, int] = {}
    for src, dsts in saltos.items():
        for d in dsts:
            if d in pos and src in pos and pos[d] <= pos[src]:
                cabezas[d] = max(cabezas.get(d, 0), pos[src])

    total = sum(len(v) for v in bloques.values())
    print(f"{ruta}: {len(bloques)} bloques basicos, {total} instrucciones\n")

    # ── lazos ────────────────────────────────────────────────────────────────────────────
    print("LAZOS (salto hacia atras = cierre de lazo)")
    print(f"  {'cabeza':<16}{'bloques':>9}{'instr':>8}{'mma':>6}{'cp.async':>10}{'fp':>6}")
    lazos = []
    for cab, ultimo in sorted(cabezas.items(), key=lambda x: pos[x[0]]):
        cuerpo = orden[pos[cab]:ultimo + 1]
        ins = [o for b in cuerpo for o in bloques[b]]
        c = collections.Counter(rol(o) for o in ins)
        lazos.append((cab, cuerpo, ins))
        print(f"  {cab:<16}{len(cuerpo):>9}{len(ins):>8}{c['mma']:>6}"
              f"{c['carga global']:>10}{c['fp']:>6}")

    # ── secciones: por rol dominante, agrupando bloques contiguos ────────────────────────
    print("\nSECCIONES (bloques contiguos agrupados por lo que hacen)")
    print(f"  {'#':>3} {'bloques':>8} {'instr':>7}  {'rol dominante':<16} detalle")
    grupos = []
    for b in orden:
        ins = bloques[b]
        if not ins:
            continue
        c = collections.Counter(rol(o) for o in ins)
        dom = max(("mma", "carga global", "shared", "bits", "fp", "div entera"),
                  key=lambda r: c.get(r, 0))
        if c.get(dom, 0) == 0:
            dom = c.most_common(1)[0][0]
        if grupos and grupos[-1][0] == dom:
            grupos[-1][1].append(b)
            grupos[-1][2].update(c)
        else:
            grupos.append([dom, [b], collections.Counter(c)])
    for i, (dom, bs, c) in enumerate(grupos):
        n = sum(c.values())
        if n < 12:
            continue
        det = " ".join(f"{k}={v}" for k, v in c.most_common(5) if k != dom)
        print(f"  {i:>3} {len(bs):>8} {n:>7}  {dom:<16} {det}")

    # ── tamaño de instruccion: anchos de acceso ──────────────────────────────────────────
    print("\nANCHOS DE ACCESO (lo que se mueve por instruccion)")
    anchos = collections.Counter()
    for b in orden:
        for o in bloques[b]:
            m = re.search(r"\.v(\d)\.[bsu](\d+)", o) or re.search(r"\.[bsu](\d+)$", o)
            if o.startswith(("ld.", "st.", "cp.async")):
                mm = re.search(r"v(\d)", o)
                bits = re.search(r"[bsuf](\d+)", o)
                v = int(mm.group(1)) if mm else 1
                bb = int(bits.group(1)) if bits else 32
                anchos[(o.split(".")[0] + "." + o.split(".")[1], v * bb)] += 1
    for (fam, bits), n in sorted(anchos.items(), key=lambda x: -x[1])[:10]:
        print(f"  {fam:<18} {bits:>4} bits x {n}")


if __name__ == "__main__":
    main()
