#!/usr/bin/env python3
"""Auditoria de punto flotante en los kernels calientes del decode.

Por que existe: ya paso varias veces que un kernel que en el fuente parece entero termina
emitiendo instrucciones de punto flotante — una division que el compilador resuelve con
reciproco fp, un `float` que se coló en un indice, una conversion de ida y vuelta. Eso no se ve
leyendo el .cu; se ve en el SASS. Esto lo cuenta para TODOS los kernels calientes de una, en vez
de ir de a uno.

Fuentes: los .so de CUDA (cuobjdump) y el PTX que Triton deja cacheado.

Uso: auditar_fp.py [--traza DIR] [--top N]
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import re
import subprocess
import sys

CUOBJDUMP = "/usr/local/cuda/bin/cuobjdump"

# Familias de instrucciones SASS. El orden importa: se prueba de arriba a abajo y gana la primera.
FAMILIAS = [
    ("mma entero",      r"^IMMA"),
    ("mma flotante",    r"^HMMA|^DMMA"),
    ("fp: conversion",  r"^F2F|^F2I|^I2F|^FRND"),
    ("fp: aritmetica",  r"^FADD|^FMUL|^FFMA|^FSET|^FSEL|^FMNMX|^HADD|^HMUL|^HFMA|^HSET|^HMNMX"
                        r"|^DADD|^DMUL|^DFMA|^MUFU|^FCHK|^FSWZADD"),
    ("entero: mult/sum", r"^IMAD|^IADD|^IABS|^ISETP|^IMNMX|^ISCADD"),
    ("entero: bits",    r"^LOP3|^SHF|^SHL|^SHR|^BREV|^FLO|^POPC|^PRMT|^BFE|^BFI"),
    ("memoria global",  r"^LDG|^STG|^LDGSTS|^RED|^ATOM"),
    ("memoria shared",  r"^LDS|^STS|^LDSM"),
    ("control/otros",   r"."),
]


def clasificar(op: str) -> str:
    for nombre, pat in FAMILIAS:
        if re.match(pat, op):
            return nombre
    return "control/otros"


def sass_de(so: str, fun: str, arch: str) -> list[str]:
    """Opcodes del kernel, en orden."""
    try:
        out = subprocess.run([CUOBJDUMP, "-sass", "-arch", arch, "-fun", fun, so],
                             capture_output=True, text=True, timeout=600).stdout
    except Exception:
        return []
    ops = []
    for linea in out.splitlines():
        m = re.match(r"\s+/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)", linea)
        if m:
            ops.append(m.group(1))
    return ops


def ptx_de(ruta: str) -> list[str]:
    """Opcodes de un .ptx de Triton. El PTX es mas legible que el SASS y ya esta en disco."""
    ops = []
    for linea in open(ruta):
        linea = linea.strip()
        if not linea or linea.startswith(("//", ".", "@", "$", "{", "}")):
            continue
        m = re.match(r"([a-z][a-z0-9_.]*)", linea)
        if m:
            ops.append(m.group(1))
    return ops


PTX_FP = re.compile(r"\.(f16|f32|f64|bf16|e4m3|e5m2)\b|^(add|sub|mul|div|fma|mad|max|min|neg|abs"
                    r"|rcp|sqrt|rsqrt|ex2|lg2|sin|cos|tanh|cvt|set|setp|selp)\.(f|rn|rz)")


def resumir_ptx(ops: list[str]) -> collections.Counter:
    c = collections.Counter()
    for o in ops:
        if o.startswith("mma") or o.startswith("wmma"):
            c["mma entero" if ".s8" in o or ".u8" in o or ".s4" in o else "mma flotante"] += 1
        elif PTX_FP.search(o):
            c["fp: aritmetica" if not o.startswith("cvt") else "fp: conversion"] += 1
        elif o.startswith(("ld.global", "st.global", "cp.async", "red.", "atom.")):
            c["memoria global"] += 1
        elif o.startswith(("ld.shared", "st.shared", "ldmatrix")):
            c["memoria shared"] += 1
        elif o.startswith(("and", "or", "xor", "shl", "shr", "not", "lop3", "bfe", "bfi", "prmt")):
            c["entero: bits"] += 1
        elif o.startswith(("add", "sub", "mul", "mad", "setp", "min", "max", "abs")):
            c["entero: mult/sum"] += 1
        else:
            c["control/otros"] += 1
    return c


def informe(nombre: str, ops, fuente: str) -> dict:
    c = ops if isinstance(ops, collections.Counter) else collections.Counter(
        clasificar(o) for o in ops)
    tot = sum(c.values())
    fp = c["fp: aritmetica"] + c["fp: conversion"] + c["mma flotante"]
    return {"nombre": nombre, "fuente": fuente, "total": tot, "fp": fp,
            "pct": 100.0 * fp / tot if tot else 0.0, "c": c}


_DEM: dict[str, str] = {}


def _demangle(nombres: list[str]) -> dict[str, str]:
    """Todos de una: llamar a c++filt por kernel tarda minutos con 2600 simbolos."""
    if not nombres:
        return {}
    out = subprocess.run(["c++filt"], input="\n".join(nombres), capture_output=True,
                         text=True).stdout.splitlines()
    return dict(zip(nombres, out))


def _buscar(indice: dict, corto: str) -> str | None:
    """El nombre de la traza viene recortado y sin `void`, asi que se compara por prefijo."""
    global _DEM
    if not _DEM:
        _DEM = _demangle(list(indice))
    clave_busq = corto.replace("void ", "").strip()
    for mang, dem in _DEM.items():
        d = dem.replace("void ", "").strip()
        if d.startswith(clave_busq[:55]):
            return mang
    return None


def kernels_calientes(traza: str, top: int) -> list[tuple[str, float, int]]:
    f = sorted(glob.glob(os.path.join(traza, "rank0*.json.gz")))
    if not f:
        return []
    d = json.load(gzip.open(f[0]))
    agg = collections.defaultdict(lambda: [0.0, 0])
    pasos = sum(1 for e in d["traceEvents"] if e.get("cat") == "user_annotation") or 1
    for e in d["traceEvents"]:
        if e.get("cat") == "kernel":
            agg[e["name"]][0] += e["dur"]
            agg[e["name"]][1] += 1
    salida = [(n, t / pasos, c // pasos) for n, (t, c) in agg.items()]
    salida.sort(key=lambda x: -x[1])
    return salida[:top]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traza", default="tests/bench/medicion/trazas/p2p_final_decode")
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--so", action="append", default=[])
    ap.add_argument("--triton", default="/home/usuario/.cache/triton")
    a = ap.parse_args()

    calientes = kernels_calientes(a.traza, a.top)
    if not calientes:
        print(f"sin traza en {a.traza}", file=sys.stderr)

    # indice: nombre mangled -> (so, arch) para los kernels de CUDA
    indice = {}
    for so in a.so:
        for arch in ("sm_86", "sm_80"):
            try:
                out = subprocess.run([CUOBJDUMP, "--list-text", so], capture_output=True,
                                     text=True, timeout=900).stdout
            except Exception:
                continue
            for linea in out.splitlines():
                m = re.search(r": x-(\S+)\.(sm_\d+)\.elf\.bin", linea)
                if m and m.group(2) == arch:
                    indice.setdefault(m.group(1), (so, arch))
            break

    # indice del PTX de Triton
    tri = {}
    for p in glob.glob(os.path.join(a.triton, "**", "*.ptx"), recursive=True):
        tri.setdefault(os.path.basename(p)[:-4], p)

    filas = []
    for nombre, us, lanz in calientes:
        corto = re.sub(r"\(.*", "", nombre).strip()
        corto = corto.replace("void ", "")
        ops, fuente = None, ""
        # 1) Triton, por nombre
        base = corto.split("(")[0].strip()
        if base in tri:
            ops, fuente = resumir_ptx(ptx_de(tri[base])), "ptx"
        else:
            # 2) CUDA, buscando el mangled que corresponda
            clave = _buscar(indice, corto)
            if clave:
                so, arch = indice[clave]
                ops, fuente = sass_de(so, clave, arch), f"sass {arch}"
        if ops is None:
            filas.append({"nombre": corto[:52], "fuente": "—", "total": 0, "fp": 0,
                          "pct": float("nan"), "c": collections.Counter(), "us": us})
            continue
        r = informe(corto[:52], ops, fuente)
        r["us"] = us
        filas.append(r)

    print(f"{'kernel':<54}{'us/paso':>9}{'fuente':>11}{'instr':>7}{'fp':>7}{'% fp':>7}")
    print("-" * 95)
    for r in filas:
        pct = "—" if r["total"] == 0 else f"{r['pct']:.1f}"
        print(f"{r['nombre']:<54}{r['us']:9.0f}{r['fuente']:>11}{r['total']:7d}{r['fp']:7d}{pct:>7}")

    print("\nDesglose de los que tienen punto flotante:")
    for r in sorted(filas, key=lambda x: -x["fp"])[:6]:
        if not r["fp"]:
            continue
        print(f"\n  {r['nombre']}  ({r['us']:.0f} us/paso, {r['pct']:.1f}% fp)")
        for fam, _ in FAMILIAS:
            if r["c"].get(fam):
                print(f"    {fam:<20} {r['c'][fam]:6d}")


if __name__ == "__main__":
    main()
