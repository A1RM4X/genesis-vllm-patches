# SPDX-License-Identifier: Apache-2.0
"""Barrido de formas sobre los super kernels PTX, sin levantar vLLM.

Por que existe
--------------
Los kernels son PTX embebido con constexpr HORNEADOS en el cubin: tile, BLOCK,
strides especializados. Un valor fuera de lo compilado no degrada, rompe — y
hasta ahora eso se descubria levantando el server, que tarda minutos y da un
`illegal memory access` asincrono que ni siquiera dice que kernel fue.

Estos tests recorren en segundos las formas que produccion realmente usa. Son
la red para la etapa de optimizacion: tocar un tile o una edicion del asm y
saber en el acto si rompio algo, sin arrancar el motor.

Clases de fallo que cubren, todas vistas de verdad en esta sesion:

* **variante faltante**: BLOCK es ``next_power_of_2(K)`` y es constexpr; con una
  sola variante compilada, un K distinto frena con "el cubin esta horneado con
  param[6]=8192 y llego 4096".
* **bucket de M sin ejercitar**: el gate original probaba M=1/20/40 y jamas
  lanzaba los tiles de prefill, que piden 72 KB de shared y necesitan
  ``cuFuncSetAttribute``.
* **memoria ilegal**: se sincroniza despues de CADA lanzamiento, asi que el
  error sale en el kernel culpable y no tres capas mas adelante.
* **no-finito**: SK-05 devuelve NaN/Inf a M>=1200 con el tile (256,128,64) —
  y el Triton original hace lo mismo, o sea que es previo a la conversion.
  Marcado xfail hasta que se arregle; sin esto el resto del barrido queda rojo
  y deja de servir como red.

Uso::

    pytest vllm/_genesis/tests/test_sk_ptx_formas.py -q          # todo
    pytest vllm/_genesis/tests/test_sk_ptx_formas.py -q -k sk06  # uno
"""

from __future__ import annotations

import importlib.util
import os

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("requiere GPU", allow_module_level=True)

_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "kernels")

# (modulo, funcion publica del GEMM, K, N) con la geometria per-rank TP=2 real.
GEMMS = [
    ("sk01_gdn_qkvz", "sk01_gdn_qkvz_gemm", 5120, 8192),
    ("sk02_gdn_out", "sk02_gemm_int8_scaled", 3072, 5120),
    ("sk03_fa_qkv", "sk03_fa_qkv_gemm", 5120, 7168),
    ("sk04_fa_o", "fa_o_int8_scaled_gemm", 3072, 5120),
    ("sk05_mlp_gateup", "sk05_gateup_gemm", 5120, 17408),
    ("sk06_mlp_down", "mlp_down_gemm", 8704, 5120),
    ("sk10_mtp_draft", "mtp_draft_fused_gemm", 5120, 7168),
]

# Un M por bucket de _CFG mas los que importan en produccion: 1 request, la
# banda de decode con MTP k=3 (4-10 requests -> M=16..40), el cruce de tile, y
# prefill.
EMES = (1, 4, 16, 20, 32, 40, 64, 127, 130, 512, 1200, 4096, 8192)

# M a partir del cual SK-05 devuelve no-finito con el tile de prefill. El bug
# es PREVIO a la conversion a PTX: el kernel Triton original hace exactamente lo
# mismo, bit a bit. Ver seccion 15 de tests.md.
SK05_ROTO_DESDE = 1200

# K reales del modelo; BLOCK = next_power_of_2(K) queda horneado en el cubin.
KS_QUANT = (1280, 3072, 5120, 7168, 8704)


def _carga(nombre):
    import sys
    if _DIR not in sys.path:
        sys.path.insert(0, _DIR)
    ruta = os.path.join(_DIR, nombre + ".py")
    spec = importlib.util.spec_from_file_location(nombre + "_t", ruta)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre + "_t"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("mod,fn,K,N", GEMMS, ids=[g[0] for g in GEMMS])
@pytest.mark.parametrize("M", EMES)
@pytest.mark.parametrize("con_shift", (False, True), ids=("sin_shift", "con_shift"))
def test_gemm_todas_las_formas(mod, fn, K, N, M, con_shift):
    """El GEMM corre y da finito en todo el rango de M que produccion usa.

    ``con_shift`` NO es decorativo: el bug que mato el arranque vivia justo en
    el cruce de shift activo con el tile de prefill (BLOCK_K=64). El indice del
    shift usaba `kb` en vez de `kb * BLOCK_K // SHIFT_BLOCK`, asi que con BK=64
    recorria el doble de filas de las que tiene la tabla y leia fuera de rango:
    `cuLaunchKernel(...tile256x128x64_shift1...): 700`. Con BK=128 coincidian y
    por eso decode nunca lo mostro. Probar shift y tile por separado no alcanza:
    hay que cruzarlos.
    """
    if mod == "sk05_mlp_gateup" and M >= SK05_ROTO_DESDE:
        pytest.xfail("SK-05 no-finito con el tile (256,128,64); bug previo a PTX")
    m = _carga(mod)
    f = getattr(m, fn)
    torch.manual_seed(0)
    b = torch.randint(-127, 127, (N, K), dtype=torch.int8, device="cuda").t()
    a = torch.randint(-127, 127, (M, K), dtype=torch.int8, device="cuda")
    asc = torch.rand(M, device="cuda") * 0.01 + 1e-3
    bsc = torch.rand(N, device="cuda") * 0.01 + 1e-3
    # fp32, NO int8: los kernels pasaron del shift diadico entero a la escala
    # por bloque en fp32. El puntero de shifts esta especializado a fp32 en el
    # cubin, asi que pasarle un int8 lee 4x fuera de rango y devuelve NaN. Es
    # exactamente la clase de fallo silencioso que este barrido existe para
    # atrapar; que la trampa haya caido sobre el propio test no la hace menos
    # real. Tiene que espejar `tools/monolitizar.py:operandos`.
    sh = (torch.rand((K // 128, (N + 127) // 128), dtype=torch.float32,
                     device="cuda") * 0.01 + 1e-3 if con_shift
          else torch.zeros((K // 128, (N + 127) // 128),
                           dtype=torch.float32, device="cuda"))
    out = f(a, b, asc, bsc, sh, None, torch.bfloat16)
    # Sincronizar ACA: un illegal memory access es asincrono y sin esto aparece
    # en otro test, o peor, en otro kernel.
    torch.cuda.synchronize()
    assert out.shape == (M, N)
    assert out.isfinite().all(), "no-finito en M=%d tile=%r" % (M, m._cfg(M, N))


@pytest.mark.parametrize("mod,fn,K,N", GEMMS, ids=[g[0] for g in GEMMS])
@pytest.mark.parametrize("M", (20, 130))
def test_gemm_con_residual(mod, fn, K, N, M):
    """Con residual el ABI del cubin tiene un param MENOS que sin el.

    ``stride_res_n`` solo se especializa a 1 cuando hay residual real; sin el
    vale 0 y viaja como param. Usar el cubin equivocado corre los punteros y da
    basura en silencio, asi que las dos formas tienen que ejercitarse.
    """
    if mod == "sk05_mlp_gateup" and M >= SK05_ROTO_DESDE:
        pytest.xfail("SK-05 no-finito con el tile de prefill")
    m = _carga(mod)
    f = getattr(m, fn)
    torch.manual_seed(0)
    b = torch.randint(-127, 127, (N, K), dtype=torch.int8, device="cuda").t()
    a = torch.randint(-127, 127, (M, K), dtype=torch.int8, device="cuda")
    asc = torch.rand(M, device="cuda") * 0.01 + 1e-3
    bsc = torch.rand(N, device="cuda") * 0.01 + 1e-3
    sh = torch.zeros((K // 128, (N + 127) // 128), dtype=torch.float32, device="cuda")
    res = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")
    out = f(a, b, asc, bsc, sh, res, torch.bfloat16)
    torch.cuda.synchronize()
    assert out.isfinite().all()


@pytest.mark.parametrize("K", KS_QUANT)
def test_quant_todos_los_K(K):
    """El quant de SK-09 tiene una variante por BLOCK; hay que cubrirlas."""
    m = _carga("sk09_norm_embed")
    torch.manual_seed(0)
    x = torch.randn((37, K), dtype=torch.bfloat16, device="cuda")
    q, s = m.quant_per_token(x)
    torch.cuda.synchronize()
    assert q.shape == x.shape
    assert q.dtype == torch.int8
    assert s.isfinite().all()


@pytest.mark.parametrize("K", KS_QUANT)
def test_quant_igual_al_triton(K):
    """Bit-exacto contra el Triton privado, que es la referencia.

    El Triton no se ejecuta en produccion: existe solo para esto.
    """
    m = _carga("sk09_norm_embed")
    torch.manual_seed(0)
    x = torch.randn((37, K), dtype=torch.bfloat16, device="cuda")
    q1, s1 = m.quant_per_token(x)
    q2, s2 = m._sk09_quant_triton(x)
    torch.cuda.synchronize()
    assert torch.equal(q1, q2)
    assert torch.equal(s1, s2)


def test_ningun_lanzamiento_fuera_del_custom_op():
    """Invariante de cableado, sin GPU.

    Un ``_QVARn(...)`` llamado directo cae dentro del forward que vLLM compila
    con fullgraph=True y mata el arranque. Ya paso: convivian dos caminos PTX y
    el publico usaba el que no era opaco a dynamo. Esto lo agarra en un segundo
    en vez de en un arranque de varios minutos.
    """
    import ast
    import glob
    import re
    malos = {}
    for ruta in sorted(glob.glob(os.path.join(_DIR, "sk[0-9]*.py"))):
        arbol = ast.parse(open(ruta).read())
        ok = [(n.lineno, n.end_lineno) for n in ast.walk(arbol)
              if isinstance(n, ast.FunctionDef)
              and (re.match(r"_q\d+_impl$", n.name) or n.name == "_lanzar")]
        fuera = ["%s:%d" % (n.func.id, n.lineno) for n in ast.walk(arbol)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and re.match(r"(_QVAR\d+(_\d+)?|_VAR_\d+)$", n.func.id)
                 and not any(a <= n.lineno <= b for a, b in ok)]
        if fuera:
            malos[os.path.basename(ruta)] = fuera
    assert not malos, "lanzamientos PTX fuera del custom op: %r" % malos
