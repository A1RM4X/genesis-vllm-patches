# SPDX-License-Identifier: Apache-2.0
"""sk_ops — los GEMM SK registrados como custom ops opacos a torch.compile.

Por qué existe este archivo
---------------------------
``Fp8LinearMethod.apply`` corre **dentro** de la región que ``torch.compile``
traza y que cudagraphs captura. Se comprobó metiendo un log con side-effect ahí:
la captura aborta con el mismo *"Assigning / modifying buffers of nn.Module
during forward pass"* que ya estaba documentado en el patch de PN110.

Eso tiene dos consecuencias para un lanzamiento de Triton hecho desde ``apply``:

1. **Triton especializa por valor.** Por cada argumento entero hornea una
   variante según si vale 1 y si es múltiplo de 16. Ese chequeo es una
   comparación sobre el valor y ``M`` viene de una dimensión dinámica que dynamo
   traza como símbolo **sin backing** (``u0``), así que no se puede resolver::

       torch._dynamo.exc.UserError: Could not guard on data-dependent
       expression Eq(u0, 1)
       Caused by: _sk01_gdn_qkvz_kernel[grid](...)

   ``@triton.jit(do_not_specialize=["M"])`` **no alcanza**: se probó y el guard
   sigue apareciendo.

2. **Elegir el tile según M tampoco se puede.** ``_cfg(M)`` hace
   ``(m > 32) + (m > 128) + ...``; cada comparación es otro guard sobre el mismo
   símbolo sin backing y falla igual. Y aunque no fallara, el resultado quedaría
   **horneado en el trazado** en vez de re-decidirse por forward — que es
   exactamente el bug que tenía el gate por M del camino caliente.

La solución es la misma que usa vLLM para sus propios kernels Triton: registrar
el lanzamiento como **custom op**. Dynamo no traza dentro de un custom op, lo
trata como una caja negra con una firma y una ``fake_impl`` para el meta device.
Adentro, ``M`` es un ``int`` de Python común y corriente: Triton especializa sin
problema y ``_cfg(M)`` elige el tile de verdad, por forward.

Esto es lo que habilita el requisito de kernels distintos para prefill y decode:
sin el custom op, la elección de tile no puede depender de M.
"""

from __future__ import annotations

import torch

from vllm._genesis.kernels.sk01_gdn_qkvz import sk01_gdn_qkvz_gemm
from vllm._genesis.kernels.sk02_gdn_out import sk02_gemm_int8_scaled
from vllm._genesis.kernels.sk03_fa_qkv import sk03_fa_qkv_gemm
from vllm._genesis.kernels.sk04_fa_o import fa_o_int8_scaled_gemm
from vllm._genesis.kernels.sk05_mlp_gateup import sk05_gateup_gemm
from vllm._genesis.kernels.sk06_mlp_down import mlp_down_gemm
from vllm._genesis.kernels.sk07_lm_head import _arange, lm_head_gemm
from vllm._genesis.kernels.sk10_mtp_draft import mtp_draft_fused_gemm

# id entero -> GEMM. Se usa un int y no un string porque el schema del custom op
# lo infiere de las anotaciones y un int mapea directo a SymInt.
SK01, SK02, SK03, SK04, SK05, SK06, SK07, SK10 = 1, 2, 3, 4, 5, 6, 7, 10

_TABLA = {
    SK01: sk01_gdn_qkvz_gemm,
    SK02: sk02_gemm_int8_scaled,
    SK03: sk03_fa_qkv_gemm,
    SK04: fa_o_int8_scaled_gemm,
    SK05: sk05_gateup_gemm,
    SK06: mlp_down_gemm,
    SK10: mtp_draft_fused_gemm,
}

SK_POR_ID: dict[str, int] = {
    "SK-01": SK01, "SK-02": SK02, "SK-03": SK03, "SK-04": SK04,
    "SK-05": SK05, "SK-06": SK06, "SK-07": SK07, "SK-10": SK10,
}


def _sk_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    shifts: torch.Tensor,
    epilogo: torch.Tensor,
    sk_id: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Despacha al GEMM SK correspondiente. Acá M ya es un int concreto."""
    if sk_id == SK07:
        return lm_head_gemm(a, b, a_scales, b_scales, shifts,
                            _arange(a.device, b.shape[1]), epilogo, out_dtype)
    return _TABLA[sk_id](a, b, a_scales, b_scales, shifts, epilogo, out_dtype)


def _sk_gemm_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    shifts: torch.Tensor,
    epilogo: torch.Tensor,
    sk_id: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Meta impl: sólo forma y dtype, sin tocar la GPU."""
    return torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)


_registrado = False


def registrar() -> bool:
    """Registra ``genesis::sk_gemm``. Idempotente."""
    global _registrado
    if _registrado:
        return True
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="genesis_sk_gemm",
        op_func=_sk_gemm,
        mutates_args=[],
        fake_impl=_sk_gemm_fake,
    )
    _registrado = True
    return True


def sk_gemm_op(a, b, a_scales, b_scales, shifts, epilogo, sk_id, out_dtype):
    """Llama al custom op ya registrado."""
    return torch.ops.vllm.genesis_sk_gemm(
        a, b, a_scales, b_scales, shifts, epilogo, sk_id, out_dtype)


__all__ = ["registrar", "sk_gemm_op", "SK_POR_ID"]
