# SPDX-License-Identifier: Apache-2.0
"""TDD para SK-12 — MLP gate_up+SiLU de prefill, cadena sin Triton.

Lo que se protege:

1. **Que no entre Triton.** El punto del kernel es que la cadena sea
   nvcc -> ptxas -> libcuda. Si alguien agrega un ``import triton`` el kernel
   deja de ser lo que dice ser.
2. **Que el launcher y el kernel no se desincronicen.** Los ``#define`` del
   ``.cu`` (BM/BN/NWARPS) y las constantes del modulo Python tienen que
   coincidir: si no, la grilla no cubre la salida y el resultado sale mal EN
   SILENCIO. Paso de verdad al pasar de BN=64 a BN=128 y lo agarro la
   validacion numerica, no el compilador.
3. **Que el kernel siga emitiendo IMMA.** Si el PTX deja de tener
   ``mma.sync...s8.s8.s32`` se perdio el 2x de los tensor cores enteros y el
   kernel no tiene razon de existir.

La correctitud numerica se valida en GPU con ``--check`` del propio modulo;
aca solo se corre si hay CUDA.
"""
from __future__ import annotations

import re

import pytest

sk12 = pytest.importorskip(
    "vllm._genesis.kernels.sk12_mlp_gateup_prefill",
    reason="modulo SK-12 no disponible")


def test_el_cu_existe_y_viaja_con_el_modulo():
    """Vive en kernels/cuda/ para entrar por el bind mount del contenedor."""
    src = sk12.fuente_cu()
    assert src.is_file()
    assert src.parent.name == "cuda"
    assert src.parent.parent.name == "kernels"


def _defines() -> dict[str, int]:
    txt = sk12.fuente_cu().read_text()
    out = {}
    for m in re.finditer(r"^#define\s+(BM|BN|BK|NWARPS|WM|WN)\s+(\d+)\s*$",
                         txt, re.M):
        out[m.group(1)] = int(m.group(2))
    return out


def test_launcher_y_kernel_no_se_desincronizan():
    """La grilla se calcula con BM/BN del Python: si difieren, sale basura."""
    d = _defines()
    assert d["BM"] == sk12._BM, (
        f"el .cu dice BM={d['BM']} y el launcher usa _BM={sk12._BM}: "
        "la grilla no va a cubrir la salida")
    assert d["BN"] == sk12._BN, (
        f"el .cu dice BN={d['BN']} y el launcher usa _BN={sk12._BN}")
    assert d["NWARPS"] == sk12._WARPS, (
        f"el .cu dice NWARPS={d['NWARPS']} y el launcher lanza "
        f"{sk12._WARPS} warps")


def test_el_reparto_de_warps_cubre_el_tile():
    """NWARPS warps de WM x WN tienen que tapar BM x BN, sin huecos ni solapes."""
    d = _defines()
    assert (d["BM"] // d["WM"]) * (d["BN"] // d["WN"]) == d["NWARPS"]
    assert d["BM"] % d["WM"] == 0 and d["BN"] % d["WN"] == 0


def test_bk_es_multiplo_de_16():
    """El fragmento WMMA consume k=16 y la carga es en vectores de 16 B."""
    d = _defines()
    assert d["BK"] % 16 == 0


def test_no_hay_triton_en_la_cadena():
    txt_py = (sk12.fuente_cu().parent.parent
              / "sk12_mlp_gateup_prefill.py").read_text()
    codigo = "\n".join(l for l in txt_py.splitlines()
                       if not l.strip().startswith("#"))
    assert "import triton" not in codigo
    assert "triton." not in codigo.split('"""')[-1]


def test_rechaza_formas_incompatibles():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("sin CUDA")
    a = torch.zeros((16, 64), dtype=torch.int8, device="cuda")
    wg = torch.zeros((8, 64), dtype=torch.int8, device="cuda")
    wu = torch.zeros((8, 32), dtype=torch.int8, device="cuda")
    s = torch.ones(16, device="cuda")
    with pytest.raises(ValueError, match="incompatibles"):
        sk12.sk12_gateup_silu_gemm(a, wg, wu, s, s, s)


def test_rechaza_k_no_multiplo_de_bk():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("sin CUDA")
    k = sk12._BK - 16   # cualquier K que no sea multiplo de BK
    a = torch.zeros((16, k), dtype=torch.int8, device="cuda")
    w = torch.zeros((8, k), dtype=torch.int8, device="cuda")
    s = torch.ones(16, device="cuda")
    with pytest.raises(ValueError, match=f"multiplo de {sk12._BK}"):
        sk12.sk12_gateup_silu_gemm(a, w, w, s, s, s)


@pytest.mark.cuda_required
def test_el_ptx_emite_tensor_cores_int8():
    """Sin mma...s8.s8.s32 el kernel perdio su razon de ser."""
    try:
        ptx = sk12.compilar_ptx()
    except (RuntimeError, FileNotFoundError) as e:
        pytest.skip(f"nvcc no disponible: {e}")
    assert re.search(r"mma\.sync[^;]*\.s32\.s8\.s8\.s32", ptx), \
        "el PTX no tiene IMMA: se perdio el 2x de los tensor cores enteros"
    assert "cp.async" in ptx, "se perdio el cp.async del doble buffer"


@pytest.mark.cuda_required
def test_correctitud_numerica():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("sin CUDA")
    assert sk12.check([(64, sk12._BK, 64), (100, 2 * sk12._BK, 70)])
