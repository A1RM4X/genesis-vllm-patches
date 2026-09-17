#!/usr/bin/env python3
"""Reemplaza en el PTX las divisiones por invariante con multiplicacion magica entera.

Por que: nvcc compila `a / b` con b variable como I2F -> MUFU.RCP -> FMUL -> F2I. MUFU tiene 4
ops/ciclo/SM contra 32 de las enteras. Los divisores de Marlin (nctaid.x=82, k_tiles, n_tiles) son
invariantes durante todo el kernel, asi que la division se puede hacer con
``mul.hi.u32`` + ``shr.u32``, que es entero puro.

Esto es el EXPERIMENTO: se hardcodean los divisores de una forma concreta para medir cuanto vale
la pena antes de tocar el fuente en C++. Si paga, se implementa bien pasando (magic, shift) desde
el host.

IMPORTANTE: los numeros de registro del PTX son POR KERNEL, asi que %r5 no es lo mismo en dos
entries distintas. La transformacion se aplica SOLO al kernel indicado, y se verifica que el
resultado numerico no cambie antes de creerle a la medicion.

Uso: magia.py <entrada.ptx> <salida.ptx> <marca_del_kernel> d_nctaid d_ktiles d_ntiles
"""
import re
import sys


RANGO = 1 << 24      # los numeradores son indices de tiles; 16M es holgado de sobra


def magico(d: int) -> tuple[int, int]:
    """(m, s) tal que  x // d == mulhi(x, m) >> (s - 32)  para todo x < RANGO.

    El metodo general de Granlund-Montgomery necesita un bit extra de correccion cuando m no
    entra en 32 bits (pasa con d=82). Como aca los numeradores son chiquitos — son indices de
    tiles, no valores de 31 bits — alcanza con buscar el s mas chico que da un m de 32 bits y
    VERIFICAR exhaustivamente en todo el rango. Sale mas simple y sin el paso de correccion.
    """
    if d <= 0:
        raise ValueError(d)
    for s in range(32, 64):
        m = (1 << s) // d + 1
        if m >= (1 << 32):
            continue
        # exacto en todo el rango? el error crece con x, asi que alcanza con mirar los bordes
        # de cada cociente mas un barrido denso al principio
        ok = all((x * m) >> s == x // d
                 for x in list(range(0, 8192))
                 + [k * d - 1 for k in range(1, RANGO // d, max(1, RANGO // d // 4096))]
                 + [k * d for k in range(1, RANGO // d, max(1, RANGO // d // 4096))]
                 + [RANGO - 1])
        if ok:
            return m, s
    raise ValueError(f"sin magia para d={d}")


def main() -> None:
    ent, sal, marca = sys.argv[1], sys.argv[2], sys.argv[3]
    divisores = [int(x) for x in sys.argv[4:]]
    mag = {d: magico(d) for d in divisores}
    for d, (m, s) in mag.items():
        print(f"  d={d:6d}  ->  mul.hi.u32 con m={m}  y  shr {s}")

    entero = open(ent).read()
    # acotar al kernel pedido: desde su .entry hasta el siguiente
    i = entero.index(marca)
    i = entero.rindex(".visible .entry", 0, i)
    j = entero.find(".visible .entry", i + 10)
    if j < 0:
        j = len(entero)
    cabeza, txt, cola = entero[:i], entero[i:j], entero[j:]
    # Cada registro divisor del PTX se conoce por su valor en esta forma; se marca a mano.
    # %r5 = %nctaid.x ; %r2 = k_tiles ; %r3 = n_tiles
    REG = {"%r5": divisores[0], "%r2": divisores[1], "%r3": divisores[2]}
    libre = [9000]

    def nuevo():
        libre[0] += 1
        return f"%r{libre[0]}"

    extra = []

    def rep_div(mo):
        dst, num, den = mo.group(2), mo.group(3), mo.group(4)
        if den not in REG:
            return mo.group(0)
        m, s = mag[REG[den]]
        t = nuevo()
        extra.append(t)
        return (f"\tmul.hi.u32 \t{t}, {num}, {m};\n"
                f"\tshr.u32 \t{dst}, {t}, {s - 32};")

    def rep_rem(mo):
        dst, num, den = mo.group(2), mo.group(3), mo.group(4)
        if den not in REG:
            return mo.group(0)
        d = REG[den]
        m, s = mag[d]
        t, q, p = nuevo(), nuevo(), nuevo()
        extra.extend([t, q, p])
        return (f"\tmul.hi.u32 \t{t}, {num}, {m};\n"
                f"\tshr.u32 \t{q}, {t}, {s - 32};\n"
                f"\tmul.lo.s32 \t{p}, {q}, {d};\n"
                f"\tsub.s32 \t{dst}, {num}, {p};")

    n0 = len(re.findall(r"\b(div|rem)\.[su]32\b", txt))
    txt = re.sub(r"\t(div)\.[su]32 \t(%r\d+), (%r\d+), (%r\d+);", rep_div, txt)
    txt = re.sub(r"\t(rem)\.[su]32 \t(%r\d+), (%r\d+), (%r\d+);", rep_rem, txt)
    n1 = len(re.findall(r"\b(div|rem)\.[su]32\b", txt))

    if extra:
        decl = "\t.reg .b32 \t" + ",".join(sorted(set(extra), key=lambda r: int(r[2:]))) + ";\n"
        # se declaran en el primer bloque de .reg que aparezca
        txt = txt.replace("\t.reg .b32", decl + "\t.reg .b32", 1)

    open(sal, "w").write(cabeza + txt + cola)
    print(f"divisiones: {n0} -> {n1}  ({n0 - n1} reemplazadas por multiplicacion magica)")


if __name__ == "__main__":
    main()
