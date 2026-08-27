# SPDX-License-Identifier: Apache-2.0
"""GEMM W8A16 para DECODE — mismo peso INT8, activacion bf16 sin cuantizar.

Por que existe
--------------
Medido (traza del profiler, 4 requests, sin MTP):

    GEMM Marlin           16.726 ms/fwd
    GEMM super kernels    16.840 ms/fwd   (+0.7%, empate)
    quant de activacion    0.648 ms/fwd
    ---------------------------------------
    total PN110           17.488 ms/fwd   -> 5% mas lento que Marlin

Los GEMM empatan. **Todo el deficit es el quant de la activacion**, y ese quant
existe solo porque el esquema es W8A8. En decode eso no compra nada: con M=4..40
el kernel espera pesos (90% del techo de DRAM en gate_up), asi que los 284 TOPS
de INT8 no tienen cómputo que saturar. Se paga una reduccion por fila a cambio
de nada.

Marlin no la paga porque es W8A16: dequantiza el peso DENTRO del mainloop y deja
la activacion en fp16. Este kernel hace lo mismo leyendo NUESTRO tensor int8, asi
que no hace falta una segunda copia del peso — que es lo que hacia inviable el
"phase dispatch" clasico (dos formatos = 24 GB sobre una placa de 24).

Reparto por fase:
  * prefill  -> GEMM INT8xINT8 (SK-01..07/10): ahi si hay computo que saturar y
                los tensor cores int8 dan 4x sobre fp16. Medido: +29%.
  * decode   -> este kernel: sin quant, sin segunda copia del peso.

Conversion int8 -> fp16 con PTX inline
--------------------------------------
fp16 tiene 10 bits de mantisa, asi que un entero de 8 bits entra exacto. Con
``u = x + 128`` en [0,255], el patron de bits ``0x6400 | u`` **es** el fp16 de
``1024 + u`` (0x6400 == 1024.0, y 1024..1279 tiene ulp=1). Entonces::

    valor = bitcast_fp16(0x6400 | (x + 128)) - 1152.0

Una instruccion entera en lugar de una conversion por elemento. Validado bit a
bit contra ``.to(tl.float16)`` en los 256 valores int8 posibles.

PENDIENTE: la version empaquetada de Marlin hace DOS valores por instruccion con
``prmt.b32`` + ``lop3.b32`` sobre un registro de 32 bits. Falta terminarla.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

SHIFT_BLOCK: int = 128

# (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages)
_CFG: tuple[tuple[int, int, int, int, int, int], ...] = (
    (16, 128, 128, 8, 4, 3),
    (16,  64, 128, 8, 4, 4),
    (32,  64, 128, 8, 4, 4),
    (64,  64, 128, 8, 4, 4),
    (128, 128, 128, 8, 8, 3),
)


def _cfg(m: int):
    return _CFG[(m > 8) + (m > 16) + (m > 32) + (m > 64)]


@triton.jit
def _int8_a_fp16(v):
    """int8 -> fp16, CUATRO por invocacion, sobre la palabra de 32 bits.

    La version ingenua ensancha cada int8 a int32 antes de convertir, y eso
    obliga al compilador a desempaquetar byte por byte: medido en el PTX, 80
    ``prmt.b32`` + 16 ``cvt.s32.s8`` por iteracion, el grueso de las 192
    instrucciones (46% del bucle) que costaba la dequantizacion.

    Aca se trabaja directo sobre los 4 int8 empaquetados::

        prmt lo, x, 0, 0x4140   // bytes 0,1 -> mitades bajas de dos half
        prmt hi, x, 0, 0x4342   // bytes 2,3
        lop3 $0, lo, 0x00800080, 0x64006400, 0xBE   // (lo ^ 0x80) | 0x6400
        lop3 $1, hi, ...
        sub.f16x2 $0, $0, 0x64806480                 // -1152.0 en ambas mitades

    6 instrucciones para 4 valores (1.5 por valor) contra 3 por valor.

    El XOR con 0x80 es el ``+128`` (en un byte, sumar 128 es invertir el bit de
    signo) y ``0x6400 | u`` son los bits de ``fp16(1024 + u)`` porque fp16 tiene
    10 bits de mantisa y 1024..1279 tiene ulp=1. LUT 0xBE = (a^b)|c.
    """
    return tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b32 lo, hi, c;
        prmt.b32  lo, $2, 0, 0x4140;
        prmt.b32  hi, $2, 0, 0x4342;
        lop3.b32  $0, lo, 0x00800080, 0x64006400, 0xBE;
        lop3.b32  $1, hi, 0x00800080, 0x64006400, 0xBE;
        mov.b32   c, 0x64806480;
        sub.f16x2 $0, $0, c;
        sub.f16x2 $1, $1, c;
        }
        """,
        constraints="=r,=r,r",
        args=[v],
        dtype=tl.float16, is_pure=True, pack=4,
    )


@triton.jit(do_not_specialize=["M"])
def _w8a16_decode_kernel(
    a_ptr, b_ptr, out_ptr, epi_ptr, b_scale_ptr, shifts_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_out_m, stride_out_n, stride_epi_m, stride_epi_n,
    stride_shift_k, stride_shift_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, SHIFT_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_in_group = GROUP_M * num_pid_n
    first_m = (pid // pid_in_group) * GROUP_M
    group_m = min(tl.cdiv(M, BLOCK_M) - first_m, GROUP_M)
    pid_m = first_m + ((pid % pid_in_group) % group_m)
    pid_n = (pid % pid_in_group) // group_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    # Escalar: BLOCK_N <= SHIFT_BLOCK, el tile cae en un solo bloque diadico.
    sh_ptrs = shifts_ptr + ((pid_n * BLOCK_N) // SHIFT_BLOCK) * stride_shift_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, K // BLOCK_K):
        # SIN .to(): la activacion ya es fp16 (el server corre --dtype float16).
        # Escribir .to(tl.float16) sobre bf16 mete 96 instrucciones por iteracion
        # (64 cvt.f32.bf16 + 32 cvt.rn.f16x2.f32), el 28% del bucle.
        a = tl.load(a_ptrs)
        b = _int8_a_fp16(tl.load(b_ptrs))
        p2 = tl.exp2(tl.load(sh_ptrs + (kb * BLOCK_K // SHIFT_BLOCK) * stride_shift_k).to(tl.float32))
        acc += tl.dot(a, b, out_dtype=tl.float32) * p2
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc = acc * tl.load(b_scale_ptr + offs_n).to(tl.float32)[None, :]
    acc += tl.load(epi_ptr + offs_m[:, None] * stride_epi_m + offs_n[None, :] * stride_epi_n).to(tl.float32)

    out_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(
        out_ptr + out_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n,
        acc.to(out_ptr.dtype.element_ty),
        mask=(out_m[:, None] < M) & (offs_n[None, :] < N),
    )


def w8a16_decode_gemm(a, b, b_scales, shifts, epilogo, out_dtype=torch.bfloat16):
    """``a`` [M,K] bf16/fp16 SIN cuantizar, ``b`` [K,N] int8 col-major."""
    M, K = a.shape
    N = b.shape[1]
    bm, bn, bk, gm, w, st = _cfg(M)
    out = torch.empty((M, N), dtype=out_dtype, device=a.device)
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    _w8a16_decode_kernel[grid](
        a, b, out, epilogo, b_scales, shifts, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        out.stride(0), out.stride(1), epilogo.stride(0), epilogo.stride(1),
        shifts.stride(0), shifts.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm, SHIFT_BLOCK=SHIFT_BLOCK,
        num_warps=w, num_stages=st,
    )
    return out


__all__ = ["w8a16_decode_gemm", "SHIFT_BLOCK"]
