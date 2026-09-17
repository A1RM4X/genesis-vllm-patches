#!/usr/bin/env python3
"""Plataforma de perfilado: corre ncu sobre un kernel y deja las metricas en una tabla.

ncu vive en el host (/opt/nvidia/nsight-compute) y el kernel corre dentro del contenedor de
vLLM, asi que se monta el profiler adentro. Hacen falta dos cosas o no lee los contadores:
--cap-add=SYS_ADMIN en el contenedor y sudo afuera (ERR_NVGPUCTRPERM si falta alguna).

Uso:  perfilar.py <kernel> [N] [K] [M]     con kernel = marlin | sk22
      perfilar.py comparar [N] [K] [M]     corre los dos y los pone lado a lado
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

REPO = "/home/usuario/Proyectos/genesis-vllm-patches"
IMG = "vllm/vllm-openai:v0.27.1"
NCU = "/opt/nvidia/nsight-compute/2025.1.1"

# Cada fila: (metrica de ncu, etiqueta, formato). Elegidas para contestar por que un kernel
# rinde distinto: cuanto del techo usa cada recurso, que lo limita y donde se traba.
METRICAS = [
    ("launch__grid_size",                                        "bloques del grid", "d"),
    ("launch__block_size",                                       "hilos por bloque", "d"),
    ("launch__waves_per_multiprocessor",                         "olas por SM", "f"),
    ("launch__registers_per_thread",                             "registros por hilo", "d"),
    ("launch__occupancy_per_register_count",                     "ocupacion max x registros", "f"),
    ("launch__occupancy_per_shared_mem_size",                    "ocupacion max x shared", "f"),
    ("sm__warps_active.avg.pct_of_peak_sustained_active",        "OCUPACION LOGRADA %", "f"),
    ("sm__throughput.avg.pct_of_peak_sustained_elapsed",         "SM del techo %", "f"),
    ("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",   "DRAM del techo %", "f"),
    ("dram__bytes.sum",                                          "bytes de DRAM", "d"),
    ("lts__t_sector_hit_rate.pct",                               "L2 aciertos %", "f"),
    ("lts__t_sectors.sum",                                       "sectores de L2", "d"),
    ("lts__throughput.avg.pct_of_peak_sustained_elapsed",        "L2 del techo %", "f"),
    ("l1tex__t_sectors.sum",                                     "sectores de L1", "d"),
    ("l1tex__throughput.avg.pct_of_peak_sustained_active",       "L1 del techo %", "f"),
    ("l1tex__t_sector_hit_rate.pct",                             "L1 aciertos %", "f"),
    ("l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",       "CONFLICTOS de banco", "d"),
    ("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",  "  conflictos en ld", "d"),
    ("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",  "  conflictos en st", "d"),
    ("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.sum", "  conflictos en ldmatrix", "d"),
    ("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",      "  frentes de onda ld", "d"),
    ("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ldsm.sum",    "  frentes de onda ldsm", "d"),
    ("smsp__warps_eligible.avg.per_cycle_active",                "WARPS ELEGIBLES x ciclo", "f"),
    ("smsp__issue_active.avg.pct_of_peak_sustained_active",      "ciclos que emite %", "f"),
    ("smsp__inst_executed.sum",                                  "instrucciones", "d"),
    ("sm__inst_executed_pipe_tensor.sum",                        "instrucciones de mma", "d"),
    ("smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct", "traba: espera memoria %", "f"),
    ("smsp__warp_issue_stalled_barrier_per_warp_active.pct",     "traba: barrera %", "f"),
    ("smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct", "traba: shared/MIO %", "f"),
    ("smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct", "traba: cola MIO llena %", "f"),
    ("gpu__time_duration.sum",                                   "tiempo (ns)", "d"),
]


def perfilar(kernel: str, n: int, k: int, m: int) -> dict[str, float]:
    # ncu tiene que correr ADENTRO del contenedor: desde el host no atraviesa el namespace de
    # docker y devuelve "No kernels were profiled".
    cmd = ["docker", "run", "--rm", "--gpus", "all", "--cap-add=SYS_ADMIN",
           "-v", f"{REPO}:/w", "-w", "/w",
           "-v", f"{REPO}/vllm/_genesis:/usr/local/lib/python3.12/dist-packages/vllm/_genesis:ro",
           "-v", "/home/usuario/.cache/genesis:/root/.cache/genesis",
           "-v", f"{NCU}:{NCU}:ro",
           "-e", "S16_SO=/root/.cache/genesis/bps1/genesis_marlin_s16.so",
           "-e", f"KERNEL={kernel}", "-e", f"N={n}", "-e", f"K={k}", "-e", f"M={m}",
           "-e", f"WARPS={os.environ.get('WARPS', '2')}", "-e", f"SK={os.environ.get('SK', '1')}", "-e", f"ETAPAS={os.environ.get('ETAPAS', '3')}",
           "--entrypoint", f"{NCU}/ncu", IMG,
           "--profile-from-start", "off", "--target-processes", "all",
           "--metrics", ",".join(x[0] for x in METRICAS), "--csv",
           "python3", "/w/tests/proto/perf/correr_kernel.py"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    vals: dict[str, float] = {}
    for ln in r.stdout.splitlines():
        campos = [c.strip('"') for c in re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', ln)]
        if len(campos) < 3:
            continue
        nombre, valor = campos[-3], campos[-1]
        if nombre in {x[0] for x in METRICAS}:
            try:
                vals[nombre] = float(valor.replace(".", "").replace(",", ".")
                                     if valor.count(",") == 1 and valor.count(".") >= 1
                                     else valor.replace(",", "."))
            except ValueError:
                pass
    if not vals:
        print(r.stdout[-3000:]); print(r.stderr[-3000:]); sys.exit(1)
    return vals


def regiones(kernel: str, n: int, k: int, m: int) -> None:
    """Metricas atribuidas a la LINEA de fuente. Pide -lineinfo (GENESIS_LINEINFO=1 para SK-22).

    Las dos que importan: cuantas instrucciones se ejecutan por linea, y cuantos ciclos de stall
    acumula cada una. La segunda es la que dice donde se va el tiempo — una linea con pocas
    instrucciones pero muchos stalls es una espera, no trabajo.
    """
    cmd = ["docker", "run", "--rm", "--gpus", "all", "--cap-add=SYS_ADMIN",
           "-v", f"{REPO}:/w", "-w", "/w",
           "-v", f"{REPO}/vllm/_genesis:/usr/local/lib/python3.12/dist-packages/vllm/_genesis:ro",
           "-v", "/home/usuario/.cache/genesis:/root/.cache/genesis",
           "-v", f"{NCU}:{NCU}:ro",
           "-e", "S16_SO=/root/.cache/genesis/bps1/genesis_marlin_s16.so",
           "-e", "GENESIS_LINEINFO=1",
           "-e", f"KERNEL={kernel}", "-e", f"N={n}", "-e", f"K={k}", "-e", f"M={m}",
           "-e", f"WARPS={os.environ.get('WARPS', '2')}", "-e", f"SK={os.environ.get('SK', '1')}", "-e", f"ETAPAS={os.environ.get('ETAPAS', '3')}",
           "--entrypoint", f"{NCU}/ncu", IMG,
           "--profile-from-start", "off", "--target-processes", "all",
           "--import-source", "yes", "--set", "source",
           "--page", "source", "--print-source", "sass",
           "python3", "/w/tests/proto/perf/correr_kernel.py"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-8000:] if r.stdout.strip() else r.stderr[-4000:])


def main() -> None:
    que = sys.argv[1] if len(sys.argv) > 1 else "comparar"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5120
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 5120
    m = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    print(f"forma {n}x{k}, M={m}\n")
    if que == "regiones":
        regiones(os.environ.get("KERNEL", "sk22"), n, k, m)
        return
    if que == "comparar":
        a, b = perfilar("marlin", n, k, m), perfilar("sk22", n, k, m)
        print(f"  {'metrica':<32}{'marlin':>14}{'sk22':>14}{'sk22/marlin':>13}")
        for met, etq, fmt in METRICAS:
            va, vb = a.get(met), b.get(met)
            if va is None or vb is None:
                continue
            rel = f"{vb / va:.2f}x" if va else "-"
            f = "{:>14,.0f}" if fmt == "d" else "{:>14.1f}"
            print(f"  {etq:<32}" + f.format(va) + f.format(vb) + f"{rel:>13}")
    else:
        v = perfilar(que, n, k, m)
        for met, etq, fmt in METRICAS:
            if met in v:
                print(f"  {etq:<32}{v[met]:>16,.1f}")


if __name__ == "__main__":
    main()
