#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Genera la version PTX monolitica de un super kernel.

Que hace
--------
Compila el ``@triton.jit`` del GEMM en TODAS las variantes que produccion
puede pedir (bucket de M x HAS_SHIFT x residual presente/ausente), saca el
PTX de cada una, y emite un .py autocontenido con:

  * los PTX embebidos como strings,
  * la plomeria de ensamblado y lanzamiento DUPLICADA adentro del archivo
    (ctypes sobre libcuda, ptxas de Triton, cuLaunchKernel),
  * el despacho por variante,
  * la API publica con la misma firma de siempre.

Sin importar ninguna plomeria compartida: cada archivo se basta solo.

Por que hace falta un generador
-------------------------------
Son ~10-20 variantes de 800-2700 lineas de PTX por kernel. Escribirlas a mano
no es viable y copiarlas a ojo es como se cuelan los desajustes de ABI.

El ABI, sin adivinar
--------------------
``handle.src.constants`` dice exactamente que posiciones especializo Triton
(los ``tl.constexpr`` declarados y los ints auto-especializados por valer 1) y
con que valor quedaron horneadas. Los params que SI viajan en el ABI son los
indices que NO estan en ``constants``, en orden, mas los dos punteros de
scratch de Triton al final (que van en NULL).

Los horneados se verifican en cada lanzamiento: si llega un valor distinto del
que quedo en el cubin el resultado seria incorrecto en silencio, asi que se
levanta ``ValueError``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import torch
import triton

RAIZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "vllm", "_genesis", "kernels")


def cargar(nombre: str):
    sys.path.insert(0, RAIZ)
    ruta = os.path.join(RAIZ, nombre + ".py")
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compilar(mod, kern, K, N, M, has_shift, con_residual):
    """Compila una variante y devuelve (handle, cfg, constants, orden_abi)."""
    bm, bn, bk, gm, warps, stages = mod._cfg(M, N)
    dev = "cuda"
    b = torch.randint(-127, 127, (N, K), dtype=torch.int8, device=dev).t()
    a = torch.randint(-127, 127, (M, K), dtype=torch.int8, device=dev)
    asc = torch.rand(M, device=dev)
    bsc = torch.rand(N, device=dev)
    out = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    if con_residual:
        res = torch.zeros((M, N), dtype=torch.bfloat16, device=dev)
        rm, rn = res.stride(0), res.stride(1)
    else:
        # mismo truco que el codigo de produccion: escalar 0-d con strides 0
        res = torch.zeros((), dtype=torch.bfloat16, device=dev)
        rm, rn = 0, 0
    sh = torch.zeros((K // 128, (N + 127) // 128), dtype=torch.int8, device=dev)
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    h = kern[grid](
        a, b, out, res, asc, bsc, sh, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        out.stride(0), out.stride(1), rm, rn,
        sh.stride(0), sh.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm, SHIFT_BLOCK=128,
        HAS_SHIFT=has_shift, num_warps=warps, num_stages=stages,
    )
    cons = {k[0]: v for k, v in h.src.constants.items()}
    n_args = len(h.src.fn.arg_names)
    orden = [i for i in range(n_args) if i not in cons]
    return h, (bm, bn, bk, gm, warps, stages), cons, orden




# --- camino condicional SK-07 (aditivo: no toca nada de lo de arriba) ---
#
# SK-07 no encaja en compilar(): tiene un parametro extra ``idx_ptr`` que
# recorta el vocabulario, llama ``BLOCK_S`` al tile de N y NO tiene el
# constexpr ``HAS_SHIFT`` (el shift diadico se aplica siempre). Ademas su eje
# de variantes no es HAS_SHIFT sino la divisibilidad por 16 de S: el vocabulario
# completo (124160) la cumple y un ``sampled_ids`` cualquiera no, y Triton usa
# ese ``tt.divisibility`` para vectorizar.


def compilar_sk07(mod, kern, cfg, K, S, con_residual, b, idx_full, M=8):
    """Compila UNA variante de SK-07. Devuelve (handle, constants, orden_abi)."""
    bm, bs_, bk, gm, warps, stages = cfg
    dev = "cuda"
    a = torch.randint(-127, 127, (M, K), dtype=torch.int8, device=dev)
    asc = torch.rand(M, device=dev)
    bsc = torch.rand(b.shape[1], device=dev)
    out = torch.empty((M, S), dtype=torch.bfloat16, device=dev)
    idx = idx_full[:S]
    if con_residual:
        res = torch.zeros((M, S), dtype=torch.bfloat16, device=dev)
        rm, rs = res.stride(0), res.stride(1)
    else:
        res = torch.zeros((), dtype=torch.bfloat16, device=dev)
        rm, rs = 0, 0
    sh = torch.zeros((K // 128, (b.shape[1] + 127) // 128),
                     dtype=torch.int8, device=dev)
    grid = (triton.cdiv(M, bm) * triton.cdiv(S, bs_),)
    h = kern[grid](
        a, b, out, idx, res, asc, bsc, sh,
        M, S, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        out.stride(0), out.stride(1), rm, rs,
        sh.stride(0), sh.stride(1),
        BLOCK_M=bm, BLOCK_S=bs_, BLOCK_K=bk, GROUP_M=gm, SHIFT_BLOCK=128,
        num_warps=warps, num_stages=stages,
    )
    cons = {k[0]: v for k, v in h.src.constants.items()}
    n_args = len(h.src.fn.arg_names)
    orden = [i for i in range(n_args) if i not in cons]
    return h, cons, orden


def recolectar_sk07(mod, kern, args):
    """Cruza los tiles distintos de la tabla x S div16 x residual."""
    dev = "cuda"
    V = args.N
    # El peso se aloca UNA vez (V*K int8 son cientos de MB) y se reusa.
    b = torch.randint(-127, 127, (V, args.K), dtype=torch.int8, device=dev).t()
    idx_full = torch.arange(V, dtype=torch.int32, device=dev)
    # Todos los tiles que _cfg() puede devolver, sin depender del muestreo de M.
    cfgs = []
    for c in tuple(mod._CFG) + (mod._CFG_POCOS_CTA,):
        if c not in cfgs:
            cfgs.append(c)
    variantes = {}
    for cfg in cfgs:
        for sdiv in (False, True):
            S = V if sdiv else 4097   # 4097: ni ==1 ni multiplo de 16
            for cr in (False, True):
                h, cons, orden = compilar_sk07(mod, kern, cfg, args.K, S, cr,
                                               b, idx_full)
                clave = (cfg, sdiv, tuple(sorted(cons.items())))
                if clave in variantes:
                    continue
                variantes[clave] = {
                    "ptx": h.asm["ptx"], "cfg": cfg, "hs": sdiv, "cons": cons,
                    "orden": orden, "warps": h.metadata.num_warps,
                    "shared": h.metadata.shared, "regs": h.n_regs,
                    "entry": h.metadata.name,
                    "scratch": h.metadata.global_scratch_size,
                    "atrs": {k[0]: val for k, val in h.src.attrs.items()},
                    "enteros": {i for i, nm in enumerate(h.src.fn.arg_names)
                                if not str(h.src.signature.get(nm, "")).startswith("*")},
                }
    del b, idx_full
    torch.cuda.empty_cache()
    return variantes


PLANTILLA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "plantillas", "plomeria_ptx.py.tpl")

CAB = '''# SPDX-License-Identifier: Apache-2.0
"""%(mod)s - bloque PTX monolitico, autocontenido.

Generado por ``tools/generar_monolitico.py``. El PTX de aca NO se edita a mano
salvo que la edicion vuelva a pasar el gate bit-exacto contra el kernel Triton
de referencia.

Este archivo no importa plomeria compartida de ningun otro modulo: el
ensamblado y el lanzamiento estan duplicados adentro, a proposito.

Variantes embebidas: %(n)d. Salen del cruce de
  * bucket de M de la tabla ``_CFG`` (el tile se elige en runtime segun M),
  * ``HAS_SHIFT`` (el Diseno B de produccion usa False),
  * residual presente o ausente.

Lo ultimo NO es cosmetico. Sin residual ``stride_res_n`` vale 0 y Triton no lo
especializa, asi que el ABI tiene UN PARAM MAS que con residual (%(abis)s).
Usar el cubin equivocado corre los punteros y da basura en silencio; el guard
de ``horneado`` lo agarra y levanta ValueError.

Geometria de esta capa: K=%(K)d, N=%(N)d.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import tempfile
import threading

import torch

'''


def emitir(args, variantes):
    """Escribe el .py monolitico con las variantes embebidas."""
    orden = sorted(variantes.items(),
                   key=lambda x: (x[1]["cfg"], x[1]["hs"], len(x[1]["orden"])))
    partes = []
    for n, (clave, v) in enumerate(orden):
        cfg, hs, _ = clave
        div16 = sorted(i for i, a in v["atrs"].items()
                       if any(x[0] == "tt.divisibility" for x in a)
                       and i in v["orden"] and i in v["enteros"])
        horn = {k: val for k, val in v["cons"].items() if k in v["enteros"]}
        partes.append((n, v, cfg, hs, div16, horn))

    abis = "/".join(str(x) for x in sorted({len(v["orden"]) for v in variantes.values()}))
    with open(args.salida, "w") as f:
        f.write(CAB % {"mod": args.mod, "n": len(partes), "abis": abis,
                       "K": args.K, "N": args.N})
        with open(PLANTILLA) as t:
            f.write(t.read())
        f.write("\n\n# --- variantes embebidas ---\n\n")
        for n, v, cfg, hs, div16, horn in partes:
            f.write('_PTX_%d = r"""%s"""\n\n' % (n, v["ptx"]))
            etiq = "sdiv" if getattr(args, "sk07", False) else "shift"
            f.write("_VAR_%d = _Nativo(\n"
                    '    "%s/tile%dx%dx%d_' + etiq + '%d_abi%d",\n'
                    '    _PTX_%d, "%s",\n'
                    "    warps=%d, shared=%d,\n"
                    "    abi=%r,\n    horneado=%r,\n    div16=%r,\n)\n\n"
                    % (n, args.mod, cfg[0], cfg[1], cfg[2], int(hs), len(v["orden"]),
                       n, v["entry"], v["warps"], v["shared"],
                       v["orden"], horn, div16))
        if getattr(args, "sk07", False):
            f.write("\n# (cfg, S_multiplo_de_16, n_params_abi) -> variante.\n"
                    "# n_params_abi distingue el caso con residual del caso sin residual.\n"
                    "_VARIANTES = {\n")
        else:
            f.write("\n# (cfg, has_shift, n_params_abi) -> variante.\n"
                    "# n_params_abi distingue el caso con residual del caso sin residual.\n"
                    "_VARIANTES = {\n")
        for n, v, cfg, hs, _d, _h in partes:
            f.write("    (%r, %r, %d): _VAR_%d,\n" % (cfg, hs, len(v["orden"]), n))
        f.write("}\n")
    print("escrito %s: %.0f KB, %d variantes"
          % (args.salida, os.path.getsize(args.salida) / 1024, len(partes)),
          file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mod", required=True, help="p.ej. sk06_mlp_down")
    ap.add_argument("--kernel", required=True, help="p.ej. _sk06_mlp_down_kernel")
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--sk07", action="store_true",
                    help="camino de SK-07: idx_ptr extra, BLOCK_S, sin HAS_SHIFT")
    args = ap.parse_args()

    mod = cargar(args.mod)
    kern = getattr(mod, args.kernel)

    if args.sk07:
        emitir(args, recolectar_sk07(mod, kern, args))
        return

    # Un M representativo por bucket de la tabla _CFG.
    MUESTRA = (8, 16, 32, 64, 128, 1024, 2048)
    variantes = {}
    for M in MUESTRA:
        for hs in (False, True):
            for cr in (False, True):
                h, cfg, cons, orden = compilar(mod, kern, args.K, args.N, M, hs, cr)
                clave = (cfg, hs, tuple(sorted(cons.items())))
                if clave in variantes:
                    continue
                variantes[clave] = {
                    "ptx": h.asm["ptx"], "cfg": cfg, "hs": hs, "cons": cons,
                    "orden": orden, "warps": h.metadata.num_warps,
                    "shared": h.metadata.shared, "regs": h.n_regs,
                    "entry": h.metadata.name, "scratch": h.metadata.global_scratch_size,
                    "atrs": {k[0]: val for k, val in h.src.attrs.items()},
                    "enteros": {i for i, nm in enumerate(h.src.fn.arg_names)
                                if not str(h.src.signature.get(nm, "")).startswith("*")},
                }
    emitir(args, variantes)
    print(f"variantes unicas: {len(variantes)}", file=sys.stderr)
    tot = sum(len(v["ptx"]) for v in variantes.values())
    print(f"PTX total: {tot/1024:.0f} KB", file=sys.stderr)
    for (cfg, hs, _), v in sorted(variantes.items(), key=lambda x: (x[1]["cfg"], x[1]["hs"])):
        print(f"  tile={cfg} shift={hs} shared={v['shared']} regs={v['regs']} "
              f"scratch={v['scratch']} abi={len(v['orden'])} horneado={sorted(v['cons'])}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
