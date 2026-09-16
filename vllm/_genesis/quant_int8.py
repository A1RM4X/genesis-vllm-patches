# SPDX-License-Identifier: Apache-2.0
"""PN134 — cuantizacion de activacion a int8 con el factor global fusionado (kernel propio PTX).

El camino W4A8 de vLLM hace dos lanzamientos por lineal:

    x_q, a_scales = per_token_quant_int8(x)     # 560 us por paso, 256 lanzamientos
    a_scales = a_scales * input_global_scale    # 272 us por paso, 208 lanzamientos

El segundo multiplica M floats (uno por token): es todo latencia de lanzamiento. ``sk19_quant``
hace las dos cosas en uno y sin punto flotante: el maximo de la fila sale de los bits del fp16
con desplazamientos, el reciproco es una division entera y la escala se arma como bits de fp32
con ``clz`` (el factor global entra como mantisa + exponente enteros).

MEDIDO EN EL SERVIDOR (perfil, decode 57k): queda en PARIDAD y por eso esta APAGADO por defecto.
    vLLM:  _per_token_quant_int8 560 us (256 lanz.) + triton_poi_fused_marlin 272 us (208) = 832
    PN134: sk19_quant            871 us (256 lanz.)
Nuestro kernel pasa de 5,8 a 3,4 us por llamada con los dos trucos enteros, pero el de vLLM esta
en 2,2: con M=5 son 5 bloques y la GPU queda parada, y ese 1,2 us de mas por llamada se come lo
que ahorra el lanzamiento que desaparece. Probado tambien con 512 y 1024 hilos por fila: igual.

Para que rinda hay que fusionarlo con el RMSNorm (la activacion se leeria una sola vez en vez de
tres): ahi los dos trucos de abajo son la base. Con M grande (prefill) el kernel de vLLM sigue
siendo mas rapido, asi que arriba de ``GENESIS_PN134_MMAX`` (256) se usa el suyo.

Se prende con ``GENESIS_ENABLE_PN134_QUANT=1``.
"""

from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("genesis.pn134")

MMAX = int(os.environ.get("GENESIS_PN134_MMAX", 256))
HILOS = int(os.environ.get("GENESIS_PN134_HILOS", 256))   # hilos por fila
_k = {}
_avisado = False


def activo() -> bool:
    return os.environ.get("GENESIS_ENABLE_PN134_QUANT", "0") == "1"


def _kernel():
    dev = torch.cuda.current_device()
    if dev not in _k:
        from vllm._genesis.kernels.ptx_lab import Kernel
        x = Kernel("sk19_quant.cu", "sk19_quant", defs=[f"-DHILOS={HILOS}"], warps=HILOS // 32)
        x.cargar()
        _k[dev] = x
    return _k[dev]


def aplicable(x: torch.Tensor, g) -> bool:
    """Se decide al TRAZAR, asi que aca no se mira el tamano del lote: con torch.compile el
    branch se hornea con la forma del trazado (M grande) y nunca se usaria en decode. El tamano
    se mira adentro del op, en ejecucion."""
    return (activo() and g is not None and x.dtype == torch.float16 and x.dim() == 2
            and (x.shape[1] & 7) == 0)


@torch.library.custom_op("genesis::pn134_quant", mutates_args=())
def _pn134_quant(x: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """int8 por token + escala con el factor global ya aplicado. Registrado como op propia para
    que sobreviva a torch.compile y quede grabado en el CUDA graph (igual que PN130)."""
    global _avisado
    m, k = x.shape
    if m > MMAX or x.stride(1) != 1:            # con M grande el kernel de vLLM es mas rapido
        from vllm import _custom_ops as ops
        xq, esc = ops.scaled_int8_quant(x)[:2]
        return xq, esc.float().reshape(m, 1) * g.float().reshape(1, 1)
    xq = torch.empty((m, k), dtype=torch.int8, device=x.device)
    esc = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    gf = g if g.dtype == torch.float32 else g.float()
    _kernel().lanzar((m, 1), [x.view(torch.int16), gf.reshape(-1).view(torch.int32),
                              xq, esc.view(torch.int32), k, x.stride(0)])
    if not _avisado:
        _avisado = True
        log.warning("PN134: cuantizacion int8 + factor global en un solo kernel (M<=%d)", MMAX)
    return xq, esc


@_pn134_quant.register_fake
def _pn134_quant_fake(x: torch.Tensor, g: torch.Tensor):
    m, k = x.shape
    return (torch.empty((m, k), dtype=torch.int8, device=x.device),
            torch.empty((m, 1), dtype=torch.float32, device=x.device))


def cuantizar(x: torch.Tensor, g: torch.Tensor):
    """Devuelve (x_int8 [M, K], escalas fp32 [M, 1]) con el factor global ya aplicado."""
    return torch.ops.genesis.pn134_quant(x, g)
