#!/usr/bin/env python3
"""Valida y mide el metodo QServe con el kernel de verdad.

Compara dos .so del mismo Marlin:
  * el de siempre, que desempaqueta con signo (|MASK, -zp, ^MASK);
  * uno compilado con GENESIS_QSERVE_CRUDO=1, donde el nibble va crudo al tensor core.

Con el segundo, el resultado del kernel es  s * suma_k a_k*q_k  con q en [0,15], o sea le SOBRA
un termino. La correccion es  8 * suma_g s_g * suma_{k in g} a_k, que es un GEMM chico de
[M, K/G] x [K/G, N]: con K/G = 40 contra K = 5120, el 0,78% del trabajo del principal.

La prueba que importa: restada la correccion, el resultado tiene que dar BIT A BIT igual al del
kernel de siempre. Si no da exacto hay un bug — no es una aproximacion.

Uso: QSERVE_SO=... BASE_SO=... qserve_kernel.py
"""
from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")

G = 128
FORMAS = [(17408, 5120), (5120, 5120), (7168, 5120), (124160, 5120)]
MES = [1, 5, 16, 64, 128, 256]


def preparar(N, K, M, dev="cuda"):
    torch.manual_seed(1234)
    b_q = torch.randint(-(2**31), 2**31 - 1, (K // 16, N * 16 // 8), dtype=torch.int32, device=dev)
    b_s = torch.randn((K // G, N), dtype=torch.float16, device=dev).abs() * 0.01 + 0.001
    m = b_s.abs().max()
    b_s16 = (b_s / m * 4096).round().to(torch.int16).view(b_s.dtype)
    ws = torch.zeros(N // 64 * 16, dtype=torch.int32, device=dev)
    x = (torch.randn(M, K, device=dev) * 24).round().clamp(-127, 127).to(torch.int8)
    a_s = torch.full((M, 1), 1 / 127, dtype=torch.float32, device=dev)
    return b_q, b_s16, ws, x, a_s, (b_s / m * 4096).round()


def correccion(x: torch.Tensor, s16_real: torch.Tensor, escala_a: float, factor: float):
    """8 * suma_g s_g * suma_{k in g} a_k, el GEMM chico.

    x es int8 [M, K]; s16_real es la escala int16 SIN empaquetar [K/G, N]. Devuelve [M, N] en la
    misma unidad en la que el kernel entrega C.
    """
    M, K = x.shape
    sumas = x.to(torch.float32).reshape(M, K // G, G).sum(2)          # [M, K/G]
    return 8.0 * (sumas @ s16_real.float()) * escala_a * factor       # [M, N]


def correr(so_path, b_q, b_s16, ws, x, a_s, M, N, K):
    from vllm.scalar_type import scalar_types
    vacio = torch.empty(0, dtype=torch.int32, device=x.device)
    ws.zero_()
    return torch.ops.genesis_marlin.marlin_gemm_s16(
        x, None, b_q, None, b_s16, a_s, None, None, vacio, vacio, ws,
        scalar_types.uint4b8.id, M, N, K, True, False, True, False)


def medir(fn, rep=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / rep * 1e6


def main() -> None:
    modo = os.environ["MODO"]            # "base" o "crudo"
    torch.ops.load_library(os.environ["S16_SO"])
    guardar = os.environ.get("GUARDAR") == "1"
    ref_dir = "/tmp/qserve"
    os.makedirs(ref_dir, exist_ok=True)

    print(f"{'forma':>16}{'M':>5}{'gemm us':>10}{'corr us':>10}{'total':>9}"
          f"{'':>3}{'exacto' if not guardar else 'ref'}")
    for N, K in FORMAS:
        for M in MES:
            b_q, b_s16, ws, x, a_s, s16_real = preparar(N, K, M)
            out = correr(None, b_q, b_s16, ws, x, a_s, M, N, K)
            t_gemm = medir(lambda: correr(None, b_q, b_s16, ws, x, a_s, M, N, K))
            clave = f"{ref_dir}/{N}_{K}_{M}.pt"
            t_corr = 0.0
            if modo == "crudo":
                # el factor que el kernel aplica a la escala int16: a_scale * (|s|max/4096) ya
                # esta adentro de C, asi que la correccion se arma en la misma unidad
                fc = 1.0
                cc = correccion(x, s16_real, 1.0 / 127, fc)
                t_corr = medir(lambda: correccion(x, s16_real, 1.0 / 127, fc))
                final = out.float() - cc
            else:
                final = out.float()
            if guardar:
                torch.save(final.cpu(), clave)
                est = "guardada"
            else:
                ref = torch.load(clave).cuda()
                d = float((final - ref).abs().max())
                est = "SI" if d == 0 else f"NO ({d:.3e})"
            print(f"{f'{N}x{K}':>16}{M:>5}{t_gemm:>10.1f}{t_corr:>10.1f}"
                  f"{t_gemm + t_corr:>9.1f}   {est}")
            del b_q, b_s16, ws, x, out
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
