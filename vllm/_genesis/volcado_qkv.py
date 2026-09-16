# SPDX-License-Identifier: Apache-2.0
"""Genesis PN123 (DIAGNOSTICO): volcado de q/k/v reales de la atencion.

Sirve para evaluar compresion de la KV (int4, codebooks, diccionarios) con los
datos de ESTE modelo antes de escribir un solo kernel. No es para produccion:
escribe a disco. Apagado salvo ``GENESIS_ENABLE_PN123_VOLCADO_QKV=1``.

Guarda, solo en el rank 0 y solo para prefills con al menos
``GENESIS_PN123_M_MIN`` tokens, las capas de ``GENESIS_PN123_CAPAS`` (indices
globales), las primeras ``GENESIS_PN123_MAX`` veces por capa. k ya tiene RoPE
aplicado: es exactamente lo que se escribe en la KV cache.

Va como custom op opaco: una llamada con efectos laterales dentro del forward
que traza dynamo se hornea o rompe el grafo (ver PN119/PN120).
"""

from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("genesis.pn123")

_ACTIVO = os.environ.get("GENESIS_ENABLE_PN123_VOLCADO_QKV", "0") == "1"
_CAPAS = {int(x) for x in os.environ.get("GENESIS_PN123_CAPAS", "3,35").split(",") if x}
_M_MIN = int(os.environ.get("GENESIS_PN123_M_MIN", "4096"))
_MAX = int(os.environ.get("GENESIS_PN123_MAX", "4"))
_DIR = os.environ.get("GENESIS_PN123_DIR", "/kv-offload/pn123_qkv")
_cuenta: dict[int, int] = {}


def activo() -> bool:
    return _ACTIVO


def _impl(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, capa: int) -> None:
    if capa not in _CAPAS or q.shape[0] < _M_MIN:
        return
    try:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
        if get_tensor_model_parallel_rank() != 0:
            return
    except Exception:
        pass
    n = _cuenta.get(capa, 0)
    if n >= _MAX:
        return
    _cuenta[capa] = n + 1
    os.makedirs(_DIR, exist_ok=True)
    f = f"{_DIR}/capa{capa:02d}_{n}.pt"
    torch.save({"q": q.detach().to(torch.float16).cpu(),
                "k": k.detach().to(torch.float16).cpu(),
                "v": v.detach().to(torch.float16).cpu()}, f)
    log.warning("[PN123] volcado %s M=%d q=%s k=%s", f, q.shape[0], tuple(q.shape), tuple(k.shape))


def _fake(q, k, v, capa) -> None:
    return None


def registrar() -> None:
    from vllm.utils.torch_utils import direct_register_custom_op
    # mutates_args=["q"] a proposito (no se muta nada): un op sin salida ni
    # mutaciones declaradas es codigo muerto para Inductor y lo borra del grafo.
    direct_register_custom_op(op_name="genesis_volcar_qkv", op_func=_impl,
                              mutates_args=["q"], fake_impl=_fake)


def volcar(q, k, v, capa: int) -> None:
    torch.ops.vllm.genesis_volcar_qkv(q, k, v, capa)


if _ACTIVO:
    try:
        registrar()
    except Exception as e:  # pragma: no cover
        log.error("[PN123] no se pudo registrar: %s", e)
