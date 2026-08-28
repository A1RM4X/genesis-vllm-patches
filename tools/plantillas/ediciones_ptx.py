# SPDX-License-Identifier: Apache-2.0
"""Ediciones bit-exactas sobre el PTX que emite Triton.

Las dos son reescrituras locales que NO cambian el resultado ni el ABI, y
estan verificadas contra el kernel Triton con ``torch.equal``. Se aplican
programaticamente: no hay PTX editado a mano en ningun lado, asi que todo el
asm del repo es reproducible desde su fuente Triton.
"""

from __future__ import annotations

import re

# Constantes fp32 potencia de dos -> su reciproco exacto. En IEEE
# ``x / 2^k == x * 2^-k`` sin error, incluidos los subnormales, asi que el
# cambio es bit-exacto.
_POW2 = {
    "0f3f800000": "0f3f800000",  # 1.0   -> 1.0
    "0f40000000": "0f3f000000",  # 2.0   -> 0.5
    "0f40800000": "0f3e800000",  # 4.0   -> 0.25
    "0f41000000": "0f3e000000",  # 8.0   -> 0.125
    "0f41800000": "0f3d800000",  # 16.0
    "0f42000000": "0f3d000000",  # 32.0
    "0f42800000": "0f3c800000",  # 64.0
    "0f43000000": "0f3c000000",  # 128.0
    "0f43800000": "0f3b800000",  # 256.0
    "0f44000000": "0f3b000000",  # 512.0
    "0f44800000": "0f3a800000",  # 1024.0
}


def editar_copysign_lop3(ptx: str) -> tuple[str, int, int]:
    """E1: ``setp.ge.f32`` + ``selp.f32`` (copysign de 0.5) -> un ``lop3.b32``.

    El redondeo del quant hace ``x + (x >= 0 ? 0.5 : -0.5)``, y Triton lo emite
    como una comparacion mas un select: dos instrucciones por elemento, y ata
    un registro de predicado. Con la LUT ``0xF8`` de ``lop3`` (``a | (b & c)``)
    sale en una: se toma el bit de signo de ``x`` y se pega sobre la magnitud
    de 0.5. Es exactamente copysign, sin tocar el valor.

    :return: ``(ptx, n_selp_reescritos, n_selp_totales)``. **Si los dos ultimos
        no coinciden hay ``selp`` sin su ``setp`` pareja y el resultado NO se
        puede usar**: significa que el patron no era el que se creia.
    """
    mapa: dict[str, str] = {}
    for m in re.finditer(
            r"^\tsetp\.ge\.f32\s+(%p\d+),\s+(%r\d+),\s+0f00000000;$",
            ptx, re.M):
        mapa[m.group(1)] = m.group(2)
    total = len(re.findall(
        r"^\tselp\.f32\s+%r\d+,\s+0f3F000000,\s+0fBF000000,\s+%p\d+;$",
        ptx, re.M))
    n = 0

    def _reemp(m: re.Match) -> str:
        nonlocal n
        src = mapa.get(m.group(2))
        if src is None:
            return m.group(0)
        n += 1
        return ("\tlop3.b32 \t%s, 0x3f000000, %s, 0x80000000, 0xF8;"
                % (m.group(1), src))

    ptx = re.sub(
        r"^\tselp\.f32\s+(%r\d+),\s+0f3F000000,\s+0fBF000000,\s+(%p\d+);$",
        _reemp, ptx, flags=re.M)
    ptx = re.sub(
        r"^\tsetp\.ge\.f32\s+%p\d+,\s+%r\d+,\s+0f00000000;\n", "", ptx,
        flags=re.M)
    return ptx, n, total


def editar_div_potencia2(ptx: str) -> tuple[str, int]:
    """E2: ``div.full.f32`` por constante potencia de dos -> ``mul.f32``.

    ``div.full.f32`` es una secuencia larga en sm_86; multiplicar por el
    reciproco exacto da el mismo bit y una sola instruccion.
    """
    regs: dict[str, str] = {}
    for m in re.finditer(r"^\tmov\.b32\s+(%r\d+),\s+(0f[0-9a-f]{8});$", ptx,
                         re.M):
        if m.group(2) in _POW2:
            regs[m.group(1)] = _POW2[m.group(2)]
    n = 0

    def _reemp(m: re.Match) -> str:
        nonlocal n
        rec = regs.get(m.group(3))
        if rec is None:
            return m.group(0)
        n += 1
        return "\tmul.f32 \t%s, %s, %s;" % (m.group(1), m.group(2), rec)

    return re.sub(r"^\tdiv\.full\.f32\s+(%r\d+),\s+(%r\d+),\s+(%r\d+);$",
                  _reemp, ptx, flags=re.M), n


def aplicar(ptx: str) -> tuple[str, int, int]:
    """Aplica E1 y E2. Devuelve ``(ptx, n_E1, n_E2)``.

    E1 se descarta entero si no cubre todos los ``selp``: mejor no editar que
    editar a medias.
    """
    p1, n1, tot1 = editar_copysign_lop3(ptx)
    e1 = n1 if (n1 > 0 and n1 == tot1) else 0
    p2, n2 = editar_div_potencia2(p1 if e1 else ptx)
    return p2, e1, n2
