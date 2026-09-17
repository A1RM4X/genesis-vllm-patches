#!/usr/bin/env python3
"""Alterna procesos entre las dos variantes de MAX_BLOQUES_POR_SM y compara las medianas.

Se alterna a nivel de PROCESO (1, 2, 1, 2, ...) en vez de correr una variante entera y despues
la otra: asi las dos ven el mismo estado de la placa. Y el reloj tiene que estar FIJO afuera
(nvidia-smi -lgc), si no se mide la rampa — ver la nota de memoria del metodo.
"""
from __future__ import annotations

import collections
import statistics
import subprocess
import sys

TANDAS = 5
IMG = "vllm/vllm-openai:v0.27.1"
REPO = "/home/usuario/Proyectos/genesis-vllm-patches"


def corre(bps: int) -> list[tuple[str, int, float]]:
    r = subprocess.run(
        ["docker", "run", "--rm", "--gpus", "all", "-v", f"{REPO}:/w", "-w", "/w",
         "-v", f"{REPO}/vllm/_genesis:/usr/local/lib/python3.12/dist-packages/vllm/_genesis:ro",
         "-v", "/home/usuario/.cache/genesis:/root/.cache/genesis",
         "-e", f"S16_SO=/root/.cache/genesis/bps{bps}/genesis_marlin_s16.so",
         "--entrypoint", "python3", IMG, "/w/tests/proto/marlin_bps.py"],
        capture_output=True, text=True)
    out = []
    for ln in r.stdout.strip().splitlines():
        if ln.count(",") == 2:
            f, m, us = ln.split(",")
            out.append((f, int(m), float(us)))
    if not out:
        print(r.stdout[-2000:], r.stderr[-2000:]); sys.exit(1)
    return out


def main() -> None:
    datos = {1: collections.defaultdict(list), 2: collections.defaultdict(list)}
    for t in range(TANDAS):
        for bps in (1, 2):
            for f, m, us in corre(bps):
                datos[bps][(f, m)].append(us)
        print(f"  tanda {t + 1}/{TANDAS} lista", flush=True)
    print(f"\n{'forma':>13}{'M':>5}{'bps=1':>9}{'bps=2':>9}{'cambio':>9}{'disp1':>8}{'disp2':>8}")
    for clave in datos[1]:
        a, b = datos[1][clave], datos[2][clave]
        ma, mb = statistics.median(a), statistics.median(b)
        da, db = (max(a) - min(a)) / min(a), (max(b) - min(b)) / min(b)
        marca = "  <-- bps=2 gana" if mb < ma * 0.98 else ("  <-- bps=1 gana" if ma < mb * 0.98 else "")
        print(f"{clave[0]:>13}{clave[1]:>5}{ma:>9.1f}{mb:>9.1f}{(ma - mb) / ma:>8.1%}"
              f"{da:>8.1%}{db:>8.1%}{marca}")


if __name__ == "__main__":
    main()
