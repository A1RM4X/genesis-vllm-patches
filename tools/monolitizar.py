#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convierte un super kernel a un bloque PTX monolitico, de punta a punta.

Hace TODO por archivo, sin edicion a mano:

  1. compila el ``@triton.jit`` del GEMM en todas las variantes que produccion
     puede pedir (bucket de M x HAS_SHIFT x residual presente/ausente),
  2. reescribe el .py: saca los ``@triton.jit``, las ``*_triton`` y el import de
     plomeria compartida; mete la plomeria DUPLICADA adentro, los PTX embebidos
     y la tabla de variantes; y reemplaza el sitio de lanzamiento por el
     despacho a la variante,
  3. valida bit-exacto contra el kernel Triton original.

El PTX se deja tal cual sale, con ``.loc``, comentarios y lineas vacias.

Uso::

    python3 tools/monolitizar.py --mod sk06_mlp_down \\
        --kernel _sk06_mlp_down_kernel --K 8704 --N 5120

Con ``--dry`` escribe a /salida y no toca el repo.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import re
import sys

import torch
import triton

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.join(os.path.dirname(AQUI), "vllm", "_genesis", "kernels")
PLANTILLA = os.path.join(AQUI, "plantillas", "plomeria_ptx.py.tpl")


def cargar(ruta: str, nombre: str):
    sys.path.insert(0, RAIZ)
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# 1. compilar todas las variantes
# --------------------------------------------------------------------------

DTYPE_A = torch.int8   # lo pisa --dtype; SK-11 (torre de vision) es bf16


def operandos(K, N, M, con_residual, dev="cuda", semilla=0):
    torch.manual_seed(semilla)
    if DTYPE_A is torch.int8:
        b = torch.randint(-127, 127, (N, K), dtype=torch.int8, device=dev).t()
        a = torch.randint(-127, 127, (M, K), dtype=torch.int8, device=dev)
    else:
        b = torch.randn((N, K), dtype=DTYPE_A, device=dev).t()
        a = torch.randn((M, K), dtype=DTYPE_A, device=dev)
    asc = torch.rand(M, device=dev) * 0.01 + 1e-3
    bsc = torch.rand(N, device=dev) * 0.01 + 1e-3
    out = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    if con_residual:
        res = torch.randn((M, N), dtype=torch.bfloat16, device=dev)
        rm, rn = res.stride(0), res.stride(1)
    else:
        res = torch.zeros((), dtype=torch.bfloat16, device=dev)
        rm, rn = 0, 0
    return a, b, out, res, asc, bsc, rm, rn


def tiene_has_shift(kern) -> bool:
    """SK-05 y SK-07 no tienen el constexpr HAS_SHIFT: su bucle aplica el shift
    siempre. Pasarselo revienta con KeyError en el empaquetado de args."""
    nombres = getattr(kern, "arg_names", None)
    if nombres is None:
        nombres = getattr(getattr(kern, "fn", None), "arg_names", [])
    return "HAS_SHIFT" in nombres


def nombres_args(kern):
    n = getattr(kern, "arg_names", None)
    if n is None:
        n = getattr(getattr(kern, "fn", None), "arg_names", [])
    return list(n)


def args_kernel(mod, K, N, M, has_shift, con_residual, semilla=0, shifts=None,
                con_hs=True, kern=None):
    """Lista COMPLETA de args en orden de declaracion del kernel.

    Se arma POR NOMBRE, no por posicion: SK-07 mete un ``idx_ptr`` extra y
    llama ``S``/``BLOCK_S`` a lo que los demas llaman ``N``/``BLOCK_N``. Armarlo
    posicionalmente funcionaba para seis kernels y fallaba en el septimo.
    """
    if kern is not None:
        return _args_por_nombre(mod, kern, K, N, M, has_shift, con_residual,
                                semilla, shifts, con_hs)
    a, b, out, res, asc, bsc, rm, rn = operandos(K, N, M, con_residual, semilla=semilla)
    if shifts is None:
        nb = (N + 127) // 128
        shifts = (torch.randint(-3, 1, (K // 128, nb), dtype=torch.int8, device="cuda")
                  if has_shift else
                  torch.zeros((K // 128, nb), dtype=torch.int8, device="cuda"))
    bm, bn, bk, gm, warps, stages = cfg_de(mod, M, N)
    return [
        a, b, out, res, asc, bsc, shifts, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        out.stride(0), out.stride(1), rm, rn,
        shifts.stride(0), shifts.stride(1),
        bm, bn, bk, gm, 128,
    ] + ([has_shift] if con_hs else []), (bm, bn, bk, gm, warps, stages), out


def cfg_de(mod, M, N):
    """``_cfg`` tiene aridad distinta segun el modulo: los INT8 son ``_cfg(m, n)``
    (usan n para la correccion por conteo de CTAs) y los W4A8 son ``_cfg(m)``."""
    try:
        return mod._cfg(M, N)
    except TypeError:
        return mod._cfg(M)


def _args_por_nombre(mod, kern, K, N, M, has_shift, con_residual, semilla,
                     shifts, con_hs):
    a, b, out, res, asc, bsc, rm, rn = operandos(K, N, M, con_residual, semilla=semilla)
    if shifts is None:
        nb = (N + 127) // 128
        shifts = (torch.randint(-3, 1, (K // 128, nb), dtype=torch.int8, device="cuda")
                  if has_shift else
                  torch.zeros((K // 128, nb), dtype=torch.int8, device="cuda"))
    bm, bn, bk, gm, warps, stages = cfg_de(mod, M, N)
    idx = torch.arange(N, dtype=torch.int32, device="cuda")
    # W4A8: peso empaquetado dos nibbles por byte -> [K//2, N], y escalas por
    # grupo de GROUP_SIZE filas de K -> [K//GROUP, N].
    grupo = getattr(mod, "GROUP_SIZE", 128)
    w_pack = torch.randint(-127, 127, (K // 2, N), dtype=torch.int8, device="cuda")
    w_sc = torch.rand((K // grupo, N), device="cuda") * 0.01 + 1e-3
    bias_v = torch.randn(N, dtype=torch.bfloat16, device="cuda")
    tabla = {
        "a_ptr": a, "b_ptr": b, "out_ptr": out, "resid_ptr": res,
        "idx_ptr": idx, "a_scale_ptr": asc, "b_scale_ptr": bsc,
        "shifts_ptr": shifts,
        "M": M, "N": N, "S": N, "K": K,
        "stride_am": a.stride(0), "stride_ak": a.stride(1),
        "stride_bk": b.stride(0), "stride_bn": b.stride(1),
        "w_ptr": w_pack, "w_scale_ptr": w_sc,
        "bias_ptr": bias_v, "stride_bias": bias_v.stride(0),
        "stride_wk": w_pack.stride(0), "stride_wn": w_pack.stride(1),
        "stride_ws_g": w_sc.stride(0), "stride_ws_n": w_sc.stride(1),
        "stride_out_m": out.stride(0), "stride_out_n": out.stride(1),
        "stride_out_s": out.stride(1),
        "stride_res_m": rm, "stride_res_n": rn, "stride_res_s": rn,
        "stride_shift_k": shifts.stride(0), "stride_shift_n": shifts.stride(1),
        "BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_S": bn, "BLOCK_K": bk,
        "GROUP_M": gm, "SHIFT_BLOCK": 128, "HAS_SHIFT": has_shift,
    }
    nombres = nombres_args(kern)
    faltan = [n for n in nombres if n not in tabla]
    if faltan:
        raise KeyError("no se como armar estos args de %s: %s"
                       % (kern.__name__ if hasattr(kern, "__name__") else kern, faltan))
    return [tabla[n] for n in nombres], (bm, bn, bk, gm, warps, stages), out


def _constexprs(kern):
    """Los constexpr que este kernel realmente declara (varian entre SKs)."""
    return [n for n in nombres_args(kern)
            if n in ("BLOCK_M", "BLOCK_N", "BLOCK_S", "BLOCK_K", "GROUP_M",
                     "SHIFT_BLOCK", "HAS_SHIFT", "SPLIT_K")]


def compilar_todas(mod, kern, K, N):
    variantes = {}
    con_hs = tiene_has_shift(kern)
    # Un M por bucket de _CFG, y NINGUNO multiplo de 16 a proposito. Con M=32
    # o M=128 Triton anota `tt.divisibility 16` sobre M y hornea esa suposicion
    # en el asm para vectorizar; el cubin resultante seria invalido para el M=40
    # real de produccion (5-10 requests con MTP k=3) y daria mal en silencio.
    # Compilando con 31/127 la suposicion no existe y la variante sirve para
    # todo el bucket. Se paga que a M multiplo de 16 el asm no vectoriza tanto
    # como lo haria el JIT: es correccion antes que velocidad.
    for M in (7, 15, 31, 63, 127, 1023, 2047):
        for hs in ((False, True) if con_hs else (False,)):
            for cr in (False, True):
                args, cfg, _ = args_kernel(mod, K, N, M, hs, cr, con_hs=con_hs, kern=kern)
                bm, bn, bk, gm, warps, stages = cfg
                grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
                ce = _constexprs(kern)
                nom = nombres_args(kern)
                pos = args[:len(nom) - len(ce)]
                kw = {n: args[nom.index(n)] for n in ce}
                h = kern[grid](*pos, **kw, num_warps=warps, num_stages=stages)
                cons = {k[0]: v for k, v in h.src.constants.items()}
                orden = [i for i in range(len(h.src.fn.arg_names)) if i not in cons]
                clave = (cfg, hs, tuple(sorted(cons.items())))
                if clave in variantes:
                    continue
                enteros = {i for i, nm in enumerate(h.src.fn.arg_names)
                           if not str(h.src.signature.get(nm, "")).startswith("*")}
                atrs = {k[0]: v for k, v in h.src.attrs.items()}
                variantes[clave] = {
                    "ptx": h.asm["ptx"], "cfg": cfg, "hs": hs, "orden": orden,
                    "warps": h.metadata.num_warps, "shared": h.metadata.shared,
                    "entry": h.metadata.name, "regs": h.n_regs,
                    "horneado": {k: v for k, v in cons.items() if k in enteros},
                    "div16": sorted(i for i, a in atrs.items()
                                    if any(x[0] == "tt.divisibility" for x in a)
                                    and i in orden and i in enteros),
                }
    return variantes


# --------------------------------------------------------------------------
# 2. reescribir el .py
# --------------------------------------------------------------------------

def transformar(fuente: str, kernel: str, bloque: str, orig=None,
                quant=()) -> tuple[str, dict]:
    """Saca Triton/plomeria compartida y mete el bloque PTX. Devuelve (src, stats)."""
    arbol = ast.parse(fuente)
    lineas = fuente.splitlines(True)
    borrar = []          # (lin_ini, lin_fin) 1-based inclusive
    reemplazos_extra = []
    # jit / triton_ref se CUENTAN pero no se borran: quedan de referencia.
    stats = {"jit": 0, "triton_ref": 0, "import_rt": 0, "ptx_viejo": 0, "lanzamiento": 0}

    for nodo in arbol.body:
        # Los @triton.jit y las *_triton NO se borran: quedan en el archivo
        # como referencia privada para los tests. El camino de PRODUCCION es el
        # PTX embebido; el Triton no lo lanza nadie salvo el gate y los pytest.
        if isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any("triton" in ast.unparse(d) and "jit" in ast.unparse(d)
                   for d in nodo.decorator_list):
                stats["jit"] += 1
            elif nodo.name.endswith("_triton"):
                stats["triton_ref"] += 1
            continue
        # import de plomeria compartida (suelto o dentro de try/except)
        elif isinstance(nodo, (ast.Import, ast.ImportFrom)):
            if "ptx_runtime" in ast.unparse(nodo):
                stats["import_rt"] += 1
                borrar.append((nodo.lineno, nodo.end_lineno))
        elif isinstance(nodo, ast.Try):
            if "ptx_runtime" in ast.unparse(nodo):
                stats["import_rt"] += 1
                borrar.append((nodo.lineno, nodo.end_lineno))
        # Descriptores KernelNativo del quant: NO se borran, se reenganchan a
        # la clase _Nativo local. Borrarlos (y borrar su PTX) dejaba a los
        # llamadores del quant referenciando nombres inexistentes: el archivo
        # compilaba y reventaba con NameError recien en runtime.
        elif isinstance(nodo, ast.Assign):
            if isinstance(nodo.value, ast.Call) and \
                    isinstance(nodo.value.func, ast.Name) and \
                    nodo.value.func.id == "KernelNativo":
                stats["ptx_viejo"] += 1
                c = nodo.value
                pos = [ast.unparse(x) for x in c.args]
                kw = {k.arg: ast.unparse(k.value) for k in c.keywords}
                var_ptx = pos[1] if len(pos) > 1 else kw.get("ptx", "")
                entry = "?"
                if orig is not None:
                    txt_ptx = getattr(orig, var_ptx.strip(), "")
                    m = re.search(r"\.visible \.entry ([A-Za-z0-9_$]+)", txt_ptx or "")
                    if m:
                        entry = m.group(1)
                abi = kw.get("idx", "[]")
                nuevo = ("%s = _Nativo(\n    %s,\n    %s, %r,\n"
                         "    warps=%s, shared=%s,\n    abi=%s,\n"
                         "    horneado=%s,\n    div16=[],\n)"
                         % (ast.unparse(nodo.targets[0]), pos[0], var_ptx, entry,
                            kw.get("num_warps", "4"), kw.get("shared", "0"),
                            abi, kw.get("horneado", "{}")))
                reemplazos_extra.append((nodo.lineno, nodo.end_lineno, 0, nuevo))

    # sitio de lanzamiento: kernel[grid](...)
    reemplazos = []
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.Call):
            continue
        f = nodo.func
        if not (isinstance(f, ast.Subscript) and isinstance(f.value, ast.Name)
                and (f.value.id == kernel or f.value.id in quant)):
            continue
        if f.value.id in quant:
            iq = list(quant).index(f.value.id)
            # El quant tiene una sola variante: se llama directo, sin tabla.
            todos = [ast.unparse(x) for x in nodo.args] + \
                [k.arg + "=" + ast.unparse(k.value) for k in nodo.keywords
                 if k.arg in ("BLOCK", "EPS")]
            simple = [ast.unparse(x) for x in nodo.args] + \
                [ast.unparse(k.value) for k in nodo.keywords
                 if k.arg in ("BLOCK", "EPS")]
            sang = " " * nodo.col_offset
            reemplazos.append((nodo.lineno, nodo.end_lineno, nodo.col_offset,
                               "_lanzar_quant%d(%s,\n%s        %s)"
                               % (iq, ast.unparse(f.slice), sang,
                                  ",\n%s        ".join([""]).join([]) or
                                  ", ".join(simple))))
            stats["lanzamiento"] += 1
            continue
        grid = ast.unparse(f.slice)
        pos = [ast.unparse(a) for a in nodo.args]
        kw = {k.arg: ast.unparse(k.value) for k in nodo.keywords}
        completos = pos + [kw[n] for n in
                           ("BLOCK_M", "BLOCK_N", "BLOCK_S", "BLOCK_K",
                            "GROUP_M", "SHIFT_BLOCK", "HAS_SHIFT",
                            "SPLIT_K") if n in kw]
        cfg = "({}, {}, {}, {}, {}, {})".format(
            kw.get("BLOCK_M"), kw.get("BLOCK_N") or kw.get("BLOCK_S"),
            kw.get("BLOCK_K"), kw.get("GROUP_M"),
            kw.get("num_warps"), kw.get("num_stages"))
        sangria = " " * (nodo.col_offset)
        nuevo = ("_lanzar({}, {}, {},\n{}        {})".format(
            grid, cfg, kw.get("HAS_SHIFT", "False"), sangria,
            ",\n{}        ".format(sangria).join(completos)))
        reemplazos.append((nodo.lineno, nodo.end_lineno, nodo.col_offset, nuevo))
        stats["lanzamiento"] += 1

    # Aplicar TODO en un solo pase de atras para adelante. Antes se hacian los
    # reemplazos primero y los borrados despues: como un reemplazo cambia la
    # cantidad de lineas, corria los numeros de linea de todo lo posterior y el
    # archivo salia picado. Solo se noto en los kernels donde el sitio de
    # lanzamiento cae ANTES de algun bloque a borrar.
    ediciones = ([(a, b, None) for a, b in borrar]
                 + [(a, b, " " * c + nuevo + "\n")
                    for a, b, c, nuevo in reemplazos + reemplazos_extra])
    for a, b, texto in sorted(ediciones, key=lambda x: -x[0]):
        lineas[a - 1:b] = [] if texto is None else [texto]
    return "".join(lineas), stats


def cabecera_imports(src: str) -> str:
    """Garantiza los imports que necesita la plomeria."""
    faltan = [m for m in ("ctypes", "os", "subprocess", "tempfile", "threading")
              if not any(l.strip() in ("import " + m,) for l in src.splitlines())]
    if not faltan:
        return src
    lineas = src.splitlines(True)
    # `from __future__` tiene que quedar primero: insertar DESPUES de el.
    fut = next((n for n, l in enumerate(lineas)
                if l.startswith("from __future__")), None)
    if fut is not None:
        i = fut + 1
    else:
        i = next((n for n, l in enumerate(lineas)
                  if l.startswith("import ") or l.startswith("from ")), 0)
    return "".join(lineas[:i] + ["import %s\n" % m for m in sorted(faltan)] + lineas[i:])


def insertar_plomeria(src: str) -> str:
    """Mete la plomeria JUSTO DESPUES de los imports, no al final.

    Los descriptores _Nativo del quant estan en el medio del modulo y se
    evaluan al importar: si la clase se define al final, el import revienta con
    NameError antes de llegar a definirla.
    """
    lineas = src.splitlines(True)
    ult = 0
    for n, l in enumerate(lineas[:200]):
        if l.startswith("import ") or l.startswith("from ") or \
                (l.startswith(("    ",)) and "import " in l):
            ult = n
    return "".join(lineas[:ult + 1]) + "\n\n" + open(PLANTILLA).read() \
        + "\n\n" + "".join(lineas[ult + 1:])


def bloque_ptx(mod_nombre, variantes) -> str:
    orden = sorted(variantes.items(),
                   key=lambda x: (x[1]["cfg"], x[1]["hs"], len(x[1]["orden"])))
    out = ["\n\n# --- variantes PTX embebidas ---\n\n"]
    tabla = []
    for n, (_clave, v) in enumerate(orden):
        cfg = v["cfg"]
        out.append('_PTX_%d = r"""%s"""\n\n' % (n, v["ptx"]))
        out.append("_VAR_%d = _Nativo(\n"
                   '    "%s/tile%dx%dx%d_shift%d_abi%d",\n'
                   '    _PTX_%d, "%s",\n    warps=%d, shared=%d,\n'
                   "    abi=%r,\n    horneado=%r,\n    div16=%r,\n)\n\n"
                   % (n, mod_nombre, cfg[0], cfg[1], cfg[2], int(v["hs"]),
                      len(v["orden"]), n, v["entry"], v["warps"], v["shared"],
                      v["orden"], v["horneado"], v["div16"]))
        tabla.append((cfg, v["hs"], n))
    out.append("\n# (cfg, has_shift) -> variantes candidatas. La eleccion final\n"
               "# entre ellas la hace _lanzar comparando los valores horneados:\n"
               "# con residual y sin residual el ABI tiene distinta cantidad de\n"
               "# params, porque stride_res_n solo se especializa cuando vale 1.\n"
               "_POR_CFG = {}\n")
    for cfg, hs, n in tabla:
        out.append("_POR_CFG.setdefault((%r, %r), []).append(_VAR_%d)\n" % (cfg, hs, n))
    out.append('''

def _lanzar(grid, cfg, has_shift, *args):
    """Elige la variante cuyos valores horneados coinciden con estos args."""
    cands = _POR_CFG.get((cfg, bool(has_shift)))
    if not cands:
        raise KeyError("no hay PTX embebido para cfg=%r has_shift=%r" % (cfg, has_shift))
    for v in cands:
        if all(args[p] == val for p, val in v.horneado.items()):
            return v((grid,) if isinstance(grid, int) else grid, *args)
    raise KeyError("no hay variante para cfg=%r has_shift=%r con estos strides" % (cfg, has_shift))
''')
    return "".join(out)


# --------------------------------------------------------------------------
# 3. gate bit-exacto
# --------------------------------------------------------------------------

def gate(orig, nuevo, kern, K, N):
    ok = tot = 0
    fallos = []
    con_hs = tiene_has_shift(kern)
    for M in (1, 20, 40):
        for hs in ((False, True) if con_hs else (False,)):
            for cr in (True, False):
                for semilla in (0, 1, 2):
                    a1, cfg, o1 = args_kernel(orig, K, N, M, hs, cr, semilla,
                                              con_hs=con_hs, kern=kern)
                    a2, _, o2 = args_kernel(orig, K, N, M, hs, cr, semilla,
                                            con_hs=con_hs, kern=kern)
                    bm, bn, bk, gm, warps, stages = cfg
                    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
                    ce = _constexprs(kern)
                    nom = nombres_args(kern)
                    kern[grid](*a1[:len(nom) - len(ce)],
                               **{n: a1[nom.index(n)] for n in ce},
                               num_warps=warps, num_stages=stages)
                    nuevo._lanzar(grid, cfg, hs, *a2)
                    torch.cuda.synchronize()
                    tot += 1
                    if torch.equal(o1, o2):
                        ok += 1
                    else:
                        d = (o1 != o2).sum().item()
                        fallos.append("M=%d shift=%s resid=%s sem=%d: %d/%d difieren"
                                      % (M, hs, cr, semilla, d, o1.numel()))
    return ok, tot, fallos


def _tabla_quant(K, M, eps, gamma=0.0, con_sp=False, semilla=0):
    """Operandos de un kernel fila-por-programa, indexados por NOMBRE.

    Cubre las variantes de quant / rmsnorm / bias+gelu de SK-01..11. Se arma
    por nombre porque las firmas no coinciden entre modulos: SK-09 suma
    ``s_pow2_ptr`` y ``GAMMA_OFFSET``, SK-11 suma ``bias_ptr``.
    """
    torch.manual_seed(semilla)
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    q = torch.empty((M, K), dtype=torch.int8, device="cuda")
    sc = torch.empty((M,), dtype=torch.float32, device="cuda")
    out = torch.empty((M, K), dtype=torch.bfloat16, device="cuda")
    w = torch.randn(K, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(K, dtype=torch.bfloat16, device="cuda")
    # s_pow2 ausente = escalar 1.0 con stride 0, igual que en produccion.
    sp = (torch.rand(K, device="cuda") + 0.5 if con_sp
          else torch.ones((), dtype=torch.float32, device="cuda"))
    return {
        "x_ptr": x, "w_ptr": w, "q_ptr": q, "s_ptr": sc, "out_ptr": out,
        "bias_ptr": bias, "s_pow2_ptr": sp,
        "K": K, "N": K, "M": M,
        "stride_xm": x.stride(0), "stride_qm": q.stride(0),
        "stride_om": out.stride(0), "stride_bias": bias.stride(0),
        "stride_sp": sp.stride(0) if con_sp else 0,
        "BLOCK": triton.next_power_of_2(K), "EPS": eps,
        "GAMMA_OFFSET": gamma,
    }


# Kernels con firma propia que no entran en la tabla generica. Devuelven
# (tabla_por_nombre, grid, nombres_de_salidas). Los dos de SK-08 MUTAN su
# buffer de estado, asi que el gate tiene que arrancar de copias identicas.
def _tabla_sk08(mod, kern, B=5, semilla=0):
    torch.manual_seed(semilla)
    nom = kern.__name__ if hasattr(kern, "__name__") else ""
    dev = "cuda"
    if "conv1d" in nom:
        D = mod.SK08_CONV_DIM
        W = mod.SK08_CONV_KERNEL_WIDTH
        x = torch.randn((B, D), dtype=torch.bfloat16, device=dev)
        w = torch.randn((D, W), dtype=torch.bfloat16, device=dev)
        bias = torch.randn(D, dtype=torch.bfloat16, device=dev)
        est = torch.randn((B, D, W), dtype=torch.bfloat16, device=dev)
        out = torch.zeros_like(x)
        t = {"x_ptr": x, "w_ptr": w, "bias_ptr": bias, "state_ptr": est,
             "out_ptr": out, "D": D,
             "stride_x_b": x.stride(0), "stride_state_b": est.stride(0),
             "stride_state_d": est.stride(1), "stride_state_w": est.stride(2),
             "stride_out_b": out.stride(0), "WIDTH": W, "BLOCK_D": 1024}
        return t, (B, triton.cdiv(D, 1024)), ["out_ptr", "state_ptr"]
    H, HV = mod.SK08_NUM_K_HEADS, mod.SK08_NUM_V_HEADS
    Kd = V = mod.SK08_HEAD_DIM
    qkv = torch.randn((B, (2 * H + HV) * Kd), dtype=torch.bfloat16, device=dev)
    a = torch.randn((B, HV), dtype=torch.float32, device=dev)
    b = torch.rand((B, HV), dtype=torch.float32, device=dev)
    A_log = torch.randn(HV, dtype=torch.float32, device=dev)
    dt_bias = torch.randn(HV, dtype=torch.float32, device=dev)
    est = torch.randn((B, HV, Kd, V), dtype=torch.float32, device=dev)
    o = torch.zeros((B, HV, V), dtype=torch.bfloat16, device=dev)
    t = {"qkv_ptr": qkv, "a_ptr": a, "b_ptr": b, "A_log_ptr": A_log,
         "dt_bias_ptr": dt_bias, "state_ptr": est, "o_ptr": o,
         "stride_qkv_b": qkv.stride(0), "stride_a_b": a.stride(0),
         "stride_state_b": est.stride(0), "stride_o_b": o.stride(0),
         "SCALE": 1.0, "H": H, "HV": HV, "K": Kd, "V": V,
         "SOFTPLUS_THRESHOLD": mod.SOFTPLUS_THRESHOLD}
    return t, (B * HV,), ["o_ptr", "state_ptr"]


def compilar_quant(mod, kern, K, eps, gamma=0.0, con_sp=False):
    """Compila el kernel de quant/rmsnorm y le aplica E1/E2.

    Una sola variante: BLOCK y EPS son constexpr fijos por capa. El guard de
    `horneado` levanta ValueError si llegan otros, y el respaldo Triton cubre
    el caso del kill-switch.
    """
    sys.path.insert(0, os.path.join(AQUI, "plantillas"))
    import ediciones_ptx as E

    nombres = nombres_args(kern)
    M = 37
    nk = getattr(kern, "__name__", "")
    grid_esp = None
    if "_sk08_" in nk:
        tabla, grid_esp, _sal = _tabla_sk08(mod, kern)
    else:
        tabla = _tabla_quant(K, M, eps, gamma, con_sp)
    faltan0 = [n for n in nombres if n not in tabla]
    if faltan0:
        raise KeyError("quant %s: no se como armar %s" % (kern.__name__, faltan0))
    args = [tabla[n] for n in nombres]
    ce = [n for n in nombres
          if str(h0.get(n, "")) == "constexpr"] if False else \
        [n for n in nombres if n in ("BLOCK", "EPS", "GAMMA_OFFSET", "WIDTH",
                                     "BLOCK_D", "SCALE", "H", "HV", "K", "V",
                                     "SOFTPLUS_THRESHOLD")]
    pos = args[:len(nombres) - len(ce)]
    nw, ns = (8, 2) if "conv1d" in nk else (4, 1) if "_sk08_" in nk else (8, 1)
    h = kern[grid_esp or (M,)](*pos, **{n: tabla[n] for n in ce},
                               num_warps=nw, num_stages=ns)
    ptx0 = h.asm["ptx"]
    ptx, e1, e2 = E.aplicar(ptx0)
    # La edicion no puede tocar el ABI: si lo movio, se descarta.
    if _params(ptx) != _params(ptx0):
        raise RuntimeError("quant: la edicion E1/E2 cambio el ABI")
    cons = {k[0]: v for k, v in h.src.constants.items()}
    orden = [i for i in range(len(nombres)) if i not in cons]
    enteros = {i for i, nm in enumerate(nombres)
               if not str(h.src.signature.get(nm, "")).startswith("*")}
    atrs = {k[0]: v for k, v in h.src.attrs.items()}
    return {
        "ptx": ptx, "entry": h.metadata.name, "warps": h.metadata.num_warps,
        "shared": h.metadata.shared, "orden": orden, "e1": e1, "e2": e2,
        "horneado": {k: v for k, v in cons.items() if k in enteros},
        "div16": sorted(i for i, a in atrs.items()
                        if any(x2[0] == "tt.divisibility" for x2 in a)
                        and i in orden and i in enteros),
        "nombres": nombres, "ce": ce, "gamma": gamma, "con_sp": con_sp,
    }


def _params(ptx):
    ini = ptx.index(".visible .entry")
    par = ptx.index("(", ini)
    return ptx[ini:ptx.index(")", par)]


def bloque_quant(mod_nombre, kernel, v, n=0) -> str:
    """Emite el PTX del quant, su descriptor y el respaldo Triton."""
    return ('\n\n# --- quant: PTX embebido (E1=%d lop3, E2=%d mul) ---\n\n'
            '_QPTX%d = r"""%s"""\n\n'
            '_QVAR%d = _Nativo(\n    "%s/%s",\n    _QPTX%d, %r,\n'
            '    warps=%d, shared=%d,\n    abi=%r,\n    horneado=%r,\n'
            '    div16=%r,\n)\n\n\n'
            'def _lanzar_quant%d(grid, *args):\n'
            '    """PTX embebido; cae al Triton si el kill-switch esta puesto.\n\n'
            '    Con ``GENESIS_PTQ_NATIVO=0`` va por el kernel Triton, que queda\n'
            '    en este archivo como referencia para los tests. Si los valores\n'
            '    horneados no coinciden (otro K, otro eps), ``_Nativo`` levanta\n'
            '    ValueError en vez de dar un numero mal en silencio.\n    """\n'
            '    if habilitado():\n'
            '        return _QVAR%d(grid, *args)\n'
            '    ce = %r\n'
            '    nom = %r\n'
            '    return %s[grid](*args[:len(nom) - len(ce)],\n'
            '                    **{n: args[nom.index(n)] for n in ce},\n'
            '                    num_warps=8, num_stages=1)\n'
            % (v["e1"], v["e2"], n, v["ptx"], n, mod_nombre, v["entry"], n,
               v["entry"], v["warps"], v["shared"], v["orden"], v["horneado"],
               v["div16"], n, n, v["ce"], v["nombres"], kernel))


def gate_sk08(orig, nuevo, kern, v, n=0):
    """Gate de los kernels con estado de SK-08.

    Los dos MUTAN su buffer de estado in-place, asi que cada corrida arranca de
    una copia identica y se comparan la salida Y el estado resultante. Comparar
    solo la salida dejaria pasar una corrupcion del estado, que es justo lo que
    se arrastraria al token siguiente.
    """
    ok = tot = 0
    fallos = []
    nombres, ce = v["nombres"], v["ce"]
    for B in (1, 5, 9):
        for semilla in (0, 1, 2):
            t1, grid, sal = _tabla_sk08(orig, kern, B, semilla)
            t2, _, _ = _tabla_sk08(orig, kern, B, semilla)
            a1 = [t1[nm] for nm in nombres]
            a2 = [t2[nm] for nm in nombres]
            kern[grid](*a1[:len(nombres) - len(ce)],
                       **{nm: t1[nm] for nm in ce},
                       num_warps=v["warps"], num_stages=1)
            getattr(nuevo, "_lanzar_quant%d" % n)(grid, *a2)
            torch.cuda.synchronize()
            tot += 1
            difs = [(nm, (t1[nm] != t2[nm]).sum().item()) for nm in sal]
            if all(d == 0 for _nm, d in difs):
                ok += 1
            else:
                fallos.append("sk08 B=%d sem=%d: %s" % (B, semilla, difs))
    return ok, tot, fallos


def gate_quant(orig, nuevo, kern, K, eps, v, n=0):
    """Bit-exacto del quant, que es donde viven las ediciones E1/E2.

    Sin esto las ediciones a mano del asm no quedarian verificadas por nada:
    son justo el lugar donde un error no se nota hasta que degrada la salida.
    """
    ok = tot = 0
    fallos = []
    nombres, ce = v["nombres"], v["ce"]
    for M in (1, 20, 37, 40):
        for semilla in (0, 1, 2):
            torch.manual_seed(semilla)
            x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
            w = torch.randn(K, dtype=torch.bfloat16, device="cuda")
            q1 = torch.zeros((M, K), dtype=torch.int8, device="cuda")
            s1 = torch.zeros((M,), dtype=torch.float32, device="cuda")
            q2 = torch.zeros_like(q1)
            s2 = torch.zeros_like(s1)
            BLOCK = triton.next_power_of_2(K)
            def arma(q, s, o):
                t = _tabla_quant(K, M, eps, v["gamma"], v["con_sp"], semilla)
                t.update({"q_ptr": q, "s_ptr": s, "out_ptr": o, "x_ptr": x,
                          "w_ptr": w})
                return [t[nm] for nm in nombres]
            # zeros, no empty: si el kernel no escribe un buffer, comparar
            # memoria sin inicializar da falsos negativos.
            o1 = torch.zeros((M, K), dtype=torch.bfloat16, device="cuda")
            o2 = torch.zeros_like(o1)
            a1, a2 = arma(q1, s1, o1), arma(q2, s2, o2)
            kern[(M,)](*a1[:len(nombres) - len(ce)],
                       **{n: a1[nombres.index(n)] for n in ce},
                       num_warps=8, num_stages=1)
            getattr(nuevo, "_lanzar_quant%d" % n)((M,), *a2)
            torch.cuda.synchronize()
            tot += 1
            # Solo se comparan las salidas que ESTE kernel declara: comparar
            # un buffer que el kernel no toca daba un falso negativo.
            pares = [(a, b) for nm, a, b in
                     (("q_ptr", q1, q2), ("s_ptr", s1, s2), ("out_ptr", o1, o2))
                     if nm in nombres]
            if all(torch.equal(a, b) for a, b in pares):
                ok += 1
            else:
                fallos.append("quant M=%d sem=%d: q difiere %d, s difiere %d"
                              % (M, semilla, (q1 != q2).sum().item(),
                                 (s1 != s2).sum().item()))
    return ok, tot, fallos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mod", required=True)
    ap.add_argument("--kernel", default="", help="GEMM; vacio = el modulo no tiene")
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--quant", default="",
                    help="kernel de quant/rmsnorm a embeber tambien (opcional)")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--dtype", default="int8", choices=("int8", "bf16"))
    ap.add_argument("--dry", default="")
    a = ap.parse_args()

    global DTYPE_A
    DTYPE_A = torch.int8 if a.dtype == "int8" else torch.bfloat16
    ruta = os.path.join(RAIZ, a.mod + ".py")
    fuente = open(ruta).read()
    orig = cargar(ruta, a.mod + "_orig")
    kern = getattr(orig, a.kernel) if a.kernel else None

    variantes = compilar_todas(orig, kern, a.K, a.N) if kern else {}
    quants = [q for q in a.quant.split(",") if q]
    qvs = [(q, compilar_quant(orig, getattr(orig, q), a.K, a.eps)) for q in quants]
    src, stats = transformar(fuente, a.kernel, "", orig, quants)
    src = insertar_plomeria(cabecera_imports(src))
    if variantes:
        src += bloque_ptx(a.mod, variantes)
    for n, (qn, qv) in enumerate(qvs):
        src += bloque_quant(a.mod, qn, qv, n)

    destino = a.dry or ruta
    # .py obligatorio: spec_from_file_location no carga otra extension
    tmp = destino[:-3] + "__tmp.py"
    open(tmp, "w").write(src)
    try:
        ast.parse(src)
    except SyntaxError as e:
        print("%-24s SINTAXIS ROTA linea %s: %s" % (a.mod, e.lineno, e.msg))
        return 1
    nuevo = cargar(tmp, a.mod + "_nuevo")
    ok, tot, fallos = gate(orig, nuevo, kern, a.K, a.N) if kern else (0, 0, [])
    for n, (qn, qv) in enumerate(qvs):
        kq = getattr(orig, qn)
        if "_sk08_" in qn:
            oq, tq, fq = gate_sk08(orig, nuevo, kq, qv, n)
        else:
            oq, tq, fq = gate_quant(orig, nuevo, kq, a.K, a.eps, qv, n)
        ok += oq
        tot += tq
        fallos += fq
    kb = len(src) / 1024
    if ok == tot:
        os.replace(tmp, destino)
        print("%-24s OK  variantes=%2d  %6.0f KB  gate %d/%d  "
              "(triton conservado: %d jit + %d ref; imp-%d ptx-%d lanz-%d)"
              % (a.mod, len(variantes), kb, ok, tot, stats["jit"],
                 stats["triton_ref"], stats["import_rt"], stats["ptx_viejo"],
                 stats["lanzamiento"]))
        return 0
    os.remove(tmp)
    print("%-24s GATE FALLA %d/%d  %s" % (a.mod, ok, tot, fallos[:3]))
    return 1


if __name__ == "__main__":
    sys.exit(main())
