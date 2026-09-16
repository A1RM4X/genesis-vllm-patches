#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Vuelca el PTX de los super kernels INT8, una variante por bucket de M.

Cierra la primera mitad del ciclo Triton -> PTX -> editar a mano -> correr. La
segunda mitad la hace ``vllm._genesis.kernels.ptx_launcher.desde_ptx``, que
ensambla el PTX editado con el ``ptxas`` que trae Triton y lo devuelve
lanzable.

Uso::

    python3 tools/volcar_ptx_sk.py --salida /ruta/ptx
    python3 tools/volcar_ptx_sk.py --sk SK-01 SK-05 --m 20 40

Hay que correrlo con GPU: Triton compila contra la capability real. El nombre
de cada archivo es ``<SK>_M<bucket>_<BM>x<BN>x<BK>_w<warps>_s<stages>.ptx``.

Ojo con el alcance: Triton compila una variante por cada combinacion de
``tl.constexpr``, y el tile se elige en RUNTIME segun M. Reemplazar un kernel
por PTX editado obliga a cargar una variante por bucket y despachar a mano.
Esto sirve para uno o dos kernels calientes, no para los 8.
"""

from __future__ import annotations

import argparse
import importlib.util
import os

import torch
import triton

# (id, archivo, funcion, K, N, tiene HAS_SHIFT, veces por forward y por rank)
KERNELS = (
    ("SK-01", "sk01_gdn_qkvz",   "_sk01_gdn_qkvz_kernel",     5120,   8192, True,  48),
    ("SK-02", "sk02_gdn_out",    "_sk02_gdn_out_int8_kernel", 3072,   5120, True,  48),
    ("SK-03", "sk03_fa_qkv",     "_sk03_fa_qkv_kernel",       5120,   7168, True,  16),
    ("SK-04", "sk04_fa_o",       "_sk04_fa_o_kernel",         3072,   5120, True,  16),
    ("SK-05", "sk05_mlp_gateup", "_sk05_mlp_gateup_kernel",   5120,  17408, False, 64),
    ("SK-06", "sk06_mlp_down",   "_sk06_mlp_down_kernel",     8704,   5120, True,  64),
    ("SK-07", "sk07_lm_head",    "_sk07_lm_head_kernel",      5120, 124160, False,  1),
    ("SK-10", "sk10_mtp_draft",  "_sk10_mtp_draft_kernel",    5120,   7168, True,   1),
)

RAIZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "vllm", "_genesis", "kernels")


def _cargar(nombre: str):
    ruta = os.path.join(RAIZ, nombre + ".py")
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _compilar(mod, kern, K, N, M, tiene_shift, es_sk07):
    """Compila una variante y devuelve el handle de Triton."""
    bm, bn, bk, gm, warps, stages = mod._cfg(M, N)
    dev = "cuda"
    b = torch.randint(-127, 127, (N, K), dtype=torch.int8, device=dev).t()
    a = torch.randint(-127, 127, (M, K), dtype=torch.int8, device=dev)
    asc = torch.rand(M, device=dev)
    bsc = torch.rand(N, device=dev)
    out = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    res = torch.zeros((M, N), dtype=torch.bfloat16, device=dev)
    sh = torch.zeros((K // 128, (N + 127) // 128), dtype=torch.int8, device=dev)
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    extra = {} if not tiene_shift else {"HAS_SHIFT": False}
    if es_sk07:
        # SK-07 recorta el vocabulario con un indice y llama BLOCK_S al tile de N.
        idx = torch.arange(N, dtype=torch.int32, device=dev)
        h = kern[grid](
            a, b, out, idx, res, asc, bsc, sh, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            out.stride(0), out.stride(1), res.stride(0), res.stride(1),
            sh.stride(0), sh.stride(1),
            BLOCK_M=bm, BLOCK_S=bn, BLOCK_K=bk, GROUP_M=gm, SHIFT_BLOCK=128,
            num_warps=warps, num_stages=stages,
        )
    else:
        h = kern[grid](
            a, b, out, res, asc, bsc, sh, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            out.stride(0), out.stride(1), res.stride(0), res.stride(1),
            sh.stride(0), sh.stride(1),
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm, SHIFT_BLOCK=128,
            num_warps=warps, num_stages=stages, **extra,
        )
    return h, (bm, bn, bk, gm, warps, stages), grid[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default="assets/ptx_sk")
    ap.add_argument("--sk", nargs="*", default=None)
    ap.add_argument("--m", nargs="*", type=int, default=[16, 20, 32, 40])
    args = ap.parse_args()

    os.makedirs(args.salida, exist_ok=True)
    print(f"{'SK':6s} {'M':>3} {'tile':>14} {'w':>2} {'s':>2} {'CTAs':>5} "
          f"{'lineas':>6} {'mma':>4} {'regs':>4} {'shared':>7} {'derr':>4}  archivo")
    for sk, arch, fn, K, N, tiene_shift, _ in KERNELS:
        if args.sk and sk not in args.sk:
            continue
        mod = _cargar(arch)
        kern = getattr(mod, fn)
        for M in args.m:
            try:
                h, cfg, ctas = _compilar(mod, kern, K, N, M, tiene_shift, sk == "SK-07")
            except Exception as e:                      # noqa: BLE001
                print(f"{sk:6s} {M:3d}  FALLO: {type(e).__name__}: {str(e)[:70]}")
                continue
            bm, bn, bk, gm, w, s = cfg
            ptx = h.asm["ptx"]
            nom = f"{sk}_M{M}_{bm}x{bn}x{bk}_w{w}_s{s}.ptx"
            with open(os.path.join(args.salida, nom), "w") as f:
                f.write(ptx)
            print(f"{sk:6s} {M:3d} {f'{bm}x{bn}x{bk}':>14} {w:2d} {s:2d} {ctas:5d} "
                  f"{len(ptx.splitlines()):6d} {ptx.count('mma.sync'):4d} {h.n_regs:4d} "
                  f"{h.metadata.shared:7d} {h.n_spills:4d}  {nom}")


if __name__ == "__main__":
    main()
