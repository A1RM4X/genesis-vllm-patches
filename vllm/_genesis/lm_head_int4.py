# SPDX-License-Identifier: Apache-2.0
"""PN139 — el ``lm_head`` en int4 por grupo, que hoy corre en fp8 (PN77).

Por que
-------
El ``lm_head`` es el segundo kernel mas caro del decode: 3,74 ms de los 28,2 del paso, en solo
5 lanzamientos (1 del modelo principal + 4 del borrador MTP con K=4). Y **esta al techo de DRAM**:
646 MB de peso fp8 por GPU leidos a 862 GB/s, contra un techo medido de 833. No hay nada que
mejorarle al kernel — la unica via es que lea menos bytes.

En int4 el mismo peso son 328 MB. Medido en ``tests/proto/banco_lmhead.py`` sobre la forma exacta
(N=124160, K=5120): **749 us contra 380 por lanzamiento**. Son 1844 us/paso si se cambian los
cinco, o 1475 si se cambia solo el borrador.

Por que el borrador primero
---------------------------
Degradar el ``lm_head`` del borrador **no puede corromper la salida**: el modelo grande verifica
cada token propuesto con su propio ``lm_head`` intacto. Lo unico que puede pasar es que baje la
tasa de aceptacion, que se mide en minutos con ``aceptacion_por_carga.py``. El del modelo
principal si toca los logits de verdad y va despues, con la suite de calidad.

La cuantizacion
---------------
Redondeo al mas cercano por grupo de 128, simetrico, sin calibracion. Para un lm_head alcanza:
no hay activaciones que propaguen el error a la siguiente capa, la salida son logits que despues
pasan por un softmax. GPTQ daria un pelin menos de error a cambio de una pasada de calibracion en
cada arranque, y no vale la pena hasta que la medicion diga que el int4 RTN molesta.

El formato de salida es el que Marlin ya consume (qweight int32 empaquetado + escalas fp16 por
grupo), asi que el GEMM es el mismo kernel que ya corre en las 256 capas del modelo.
"""

from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("genesis.pn139")

G = 128                       # group_size, el mismo del checkpoint
MARCA = "_genesis_pn139_listo"


def activo() -> bool:
    return os.environ.get("GENESIS_ENABLE_PN139_LM_HEAD_INT4", "0") == "1"


def solo_borrador() -> bool:
    """Por defecto solo el borrador: es el cambio que no puede tocar la calidad de salida."""
    return os.environ.get("GENESIS_PN139_SOLO_BORRADOR", "1") == "1"


def _es_del_borrador(prefijo: str) -> bool:
    return prefijo.startswith("mtp") or ".mtp" in prefijo or "draft" in prefijo


def cuantizar_int4_grupo(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``w`` [N, K] fp16 -> (qweight empaquetado, escalas [K/G, N]) en el formato de GPTQ.

    Simetrico con offset 8 (``uint4b8``), que es lo que espera Marlin: el valor guardado es
    ``q + 8`` en [0, 15] y el kernel le resta 8.
    """
    n, k = w.shape
    if k % G:
        raise ValueError(f"K={k} no es multiplo de {G}")
    wf = w.float().reshape(n, k // G, G)
    # escala por grupo desde el maximo absoluto; 7 y no 8 para que el negativo extremo no sature
    esc = wf.abs().amax(dim=2).clamp_min(1e-8) / 7.0            # [N, K/G]
    q = (wf / esc.unsqueeze(2)).round_().clamp_(-8, 7).to(torch.int8) + 8   # [N, K/G, G] en [0,15]
    q = q.reshape(n, k).to(torch.int32)

    # empaquetado GPTQ: 8 valores de 4 bits por int32, a lo largo de K, con K en las filas
    qt = q.t().contiguous()                                      # [K, N]
    qp = torch.zeros((k // 8, n), dtype=torch.int32, device=w.device)
    for i in range(8):
        qp |= qt[i::8] << (4 * i)
    return qp, esc.t().contiguous().to(w.dtype)                  # escalas [K/G, N]


def preparar(layer, prefijo: str = "") -> bool:
    """Cuantiza el peso del ``lm_head`` y lo deja en el layout que Marlin consume.

    Devuelve False (sin tocar nada) si no corresponde, para que el llamador siga por su camino.
    """
    if getattr(layer, MARCA, False):
        return True
    w = getattr(layer, "weight", None)
    if w is None or w.dim() != 2 or w.dtype not in (torch.float16, torch.bfloat16):
        return False
    if solo_borrador() and not _es_del_borrador(prefijo):
        return False

    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_permute_scales,
        marlin_make_workspace_new,
    )
    from vllm.model_executor.utils import replace_parameter
    from vllm.scalar_type import scalar_types
    from vllm import _custom_ops as ops

    n, k = w.shape
    dev = w.device
    qp, esc = cuantizar_int4_grupo(w.data)
    # repack al layout interno de Marlin y permutacion de las escalas, igual que el camino GPTQ
    qm = ops.gptq_marlin_repack(qp, torch.empty(0, dtype=torch.int32, device=dev),
                                k, n, 4)
    em = marlin_permute_scales(esc, size_k=k, size_n=n, group_size=G)

    replace_parameter(layer, "weight", qm)
    layer.register_parameter(
        "weight_scale", torch.nn.Parameter(em, requires_grad=False))
    layer.workspace = marlin_make_workspace_new(dev)
    layer.orig_dtype = w.dtype
    layer.pn139_n, layer.pn139_k = n, k
    layer.pn139_tipo = scalar_types.uint4b8
    setattr(layer, MARCA, True)
    log.warning("PN139: lm_head %s a int4 g%d — %.0f MB -> %.0f MB por GPU",
                prefijo or "?", G, n * k * 2 / 1e6, n * k / 2 / 1e6)
    return True


def aplicar(layer, x: torch.Tensor, bias=None) -> torch.Tensor:
    """El GEMM, por el mismo Marlin int4 que ya corre en las 256 capas del modelo."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_gptq_marlin_linear,
    )
    vacio = torch.empty(0, dtype=torch.int32, device=x.device)
    return apply_gptq_marlin_linear(
        input=x, weight=layer.weight, weight_scale=layer.weight_scale,
        weight_zp=vacio, g_idx=vacio, g_idx_sort_indices=vacio,
        workspace=layer.workspace, wtype=layer.pn139_tipo,
        input_size_per_partition=layer.pn139_k,
        output_size_per_partition=layer.pn139_n,
        is_k_full=True, bias=bias)
