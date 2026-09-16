# SPDX-License-Identifier: Apache-2.0
"""PN133 — las lineales del modulo MTP (el borrador) en W8A8.

Por que
-------
El checkpoint deja todo ``mtp.*`` fuera de la cuantizacion (esta en la lista de ``ignore``), asi
que mientras el modelo corre en W4A8 por Marlin, el borrador hace sus GEMM en fp16 por cuBLAS,
una vez por token propuesto. Medido: 2,26 ms de los 29,6 ms del paso de decode (7,6 %), con
20 lanzamientos por paso.

Que hace
--------
Despues de cargar los pesos, cuantiza cada lineal del modulo a int8 con escala por canal de
salida y reemplaza el metodo de la capa por uno que cuantiza la activacion por token y llama al
GEMM int8 de cutlass (el mismo que usa el camino W8A8 de vLLM). La mitad de bytes por peso.

Por que W8A8 y no W4
--------------------
arXiv 2505.22179 (*Speculative Decoding Meets Quantization*) mide que GPTQ sobre el borrador
degrada fuerte la tasa de aceptacion y que W4 es incompatible con la verificacion; W8A8 si es
compatible. Igual hay que medir la aceptacion antes y despues: es la unica metrica del borrador.

Se apaga con ``GENESIS_ENABLE_PN133_MTP_INT8=0``.
"""

from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("genesis.pn133")

MIN_PARAMS = int(os.environ.get("GENESIS_PN133_MIN", 4 << 20))   # no vale la pena en matrices chicas
_hechas = set()


def activo() -> bool:
    return os.environ.get("GENESIS_ENABLE_PN133_MTP_INT8", "0") == "1"


class _MetodoInt8:
    """Reemplaza el apply() de la capa: activacion int8 por token + GEMM int8 de cutlass."""

    def __init__(self, original):
        self.original = original

    def __getattr__(self, n):          # todo lo demas (create_weights, embedding, etc.) al original
        return getattr(self.original, n)

    def apply(self, layer, x, bias=None):
        # vLLM comparte una misma instancia de quant_method entre capas: si esta no se cuantizo,
        # se va por el camino original (si no, revienta con 'no attribute g133_w').
        if getattr(layer, "g133_w", None) is None:
            return self.original.apply(layer, x, bias)
        from vllm import _custom_ops as ops
        forma = x.shape
        x2 = x.reshape(-1, forma[-1])
        xq, xs, _ = ops.scaled_int8_quant(x2)
        out = ops.cutlass_scaled_mm(xq, layer.g133_w, scale_a=xs, scale_b=layer.g133_s,
                                    out_dtype=x.dtype, bias=bias)
        return out.reshape(*forma[:-1], out.shape[-1])


def _cuantizar_capa(capa) -> int:
    w = getattr(capa, "weight", None)
    if w is None or w.dim() != 2 or w.dtype not in (torch.float16, torch.bfloat16):
        return 0
    if w.numel() < MIN_PARAMS or getattr(capa, "g133_w", None) is not None:
        return 0
    met = getattr(capa, "quant_method", None)
    if met is None or isinstance(met, _MetodoInt8):
        return 0
    with torch.no_grad():
        esc = w.abs().amax(dim=1).clamp_min(1e-8).to(torch.float32) / 127.0     # por canal de salida
        q = torch.round(w.to(torch.float32) / esc[:, None]).clamp_(-127, 127).to(torch.int8)
        capa.g133_w = q.contiguous().t()          # cutlass quiere B como [K, N] COLUMNA-major
        capa.g133_s = esc.contiguous()
        capa.weight = torch.nn.Parameter(torch.empty(0, dtype=w.dtype, device=w.device),
                                         requires_grad=False)   # libera el fp16
    capa.quant_method = _MetodoInt8(met)
    return q.numel()


def cuantizar(modelo) -> None:
    """Recorre el modulo MTP y cuantiza sus lineales. Idempotente."""
    if not activo() or id(modelo) in _hechas:
        return
    _hechas.add(id(modelo))
    total = capas = 0
    for nombre, m in modelo.named_modules():
        if "lm_head" in nombre or "embed" in nombre or "shared_head" in nombre:
            continue                              # la cabeza y las embeddings no se tocan
        n = _cuantizar_capa(m)
        if n:
            total += n
            capas += 1
    if capas:
        torch.cuda.empty_cache()
        log.warning("PN133: %d lineales del borrador MTP en int8 (%.0f MiB de pesos, antes %.0f en fp16)",
                    capas, total / 2**20, 2 * total / 2**20)
