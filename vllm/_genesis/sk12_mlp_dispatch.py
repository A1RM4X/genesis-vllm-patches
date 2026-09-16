# SPDX-License-Identifier: Apache-2.0
"""PN119 — runtime: despacha el MLP de prefill por SK-12 (gate_up+SiLU fusionado).

Por que hace falta un parche aparte de PN118
--------------------------------------------
PN118 engancha a nivel de **Linear**, donde `gate_up_proj` tiene que devolver
`[M, 2N]` sin activar. SK-12 fusiona el GEMM con `SiLU(gate)*up` y devuelve
`[M, N]`: no encaja en esa firma. Por eso el despacho va al nivel del **modulo
MLP**, donde el gate_up y la activacion son la misma operacion.

Reutiliza los pesos que PN118 ya dejo convertidos en `gate_up_proj`, asi que no
duplica ni memoria ni trabajo de carga: solo cambia quien ejecuta el forward.

Solo prefill
------------
SK-12 esta dimensionado para M grande (tile 128x64, 4 warps). En decode M vale
4-40 y pierde por goleada, asi que debajo de `GENESIS_PN119_M_MIN` (default
512) el forward original queda intacto.

Layout
------
PN118 deja `w8` de `[2N, K]` int8 contiguo y `esc` de `[2N]`. En un
MergedColumnParallelLinear las filas `0..N-1` son gate y `N..2N-1` son up, asi
que los slices salen contiguos y SK-12 los consume directo.
"""

from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("genesis.pn119")

_TRUTHY = ("1", "true", "yes", "on")


def _flag(n: str, d: str = "0") -> bool:
    return os.environ.get(n, d).strip().lower() in _TRUTHY


# Se leen al IMPORTAR, no por forward: `gate_up_silu` corre dentro del grafo
# que traza dynamo, y leer os.environ ahi mete guards y ruido.
_ACTIVO = _flag("GENESIS_ENABLE_PN119_SK12_MLP")
try:
    _M_MIN = int(os.environ.get("GENESIS_PN119_M_MIN", "512"))
except ValueError:
    _M_MIN = 512


def activo() -> bool:
    return _ACTIVO


def _m_min() -> int:
    return _M_MIN


# ─────────────────────── custom op opaco a dynamo ───────────────────────
#
# El forward del MLP corre DENTRO de la region que torch.compile traza. El
# lanzador nativo de SK-12 usa un lock para ensamblar el cubin una sola vez, y
# ese camino esta marcado con @torch.compiler.disable — dynamo no lo puede
# inlinear y aborta con "Skip inlining `torch.compiler.disable()`d function".
#
# La salida es la misma que usa sk_ops.py para los SK viejos: registrar el
# lanzamiento como custom op, asi dynamo lo trata como una caja negra y ni
# intenta entrar.

_registrado = False


# Contador de invocaciones REALES. Los contadores de `gate_up_silu` no sirven
# para esto: ese codigo lo traza dynamo, asi que se ejecutan una vez al compilar
# y nunca mas. El custom op es opaco, o sea que ESTO si corre por forward — y es
# el unico lugar del camino donde se puede loguear sin romper el trazado.
_LLAMADAS = [0]


def _sk12_op(a_i8: torch.Tensor, w8: torch.Tensor, esc: torch.Tensor,
             sa: torch.Tensor) -> torch.Tensor:
    """`SiLU(gate)*up`, eligiendo el GEMM POR TALLA EN RUNTIME.

    La eleccion tiene que vivir ACA y no en el forward del MLP. El forward lo
    traza dynamo una sola vez (con el M del profile_run, 8192), asi que un
    `if M < M_MIN` alla arriba se hornea en el grafo y deja de ser un chequeo:
    medido, el grafo terminaba llamando a SK-12 con M=1188, donde PIERDE contra
    cutlass. Ganaba en los chunks grandes, perdia en los chicos y el neto daba
    exactamente cero.

    El custom op es opaco a dynamo, o sea que esto si corre por forward.
    """
    from vllm import _custom_ops as ops
    from vllm._genesis.kernels.sk12_mlp_gateup_prefill import sk12_gateup_silu_gemm

    n = w8.shape[0] // 2
    m = a_i8.shape[0]
    _LLAMADAS[0] += 1
    if _DEBUG_CADA and _LLAMADAS[0] % _DEBUG_CADA == 0:
        log.warning("[PN119] llamada %d: M=%d N=%d -> %s",
                    _LLAMADAS[0], m, n, "SK-12" if m >= _m_min() else "cutlass")

    if m < _m_min():
        # Debajo del piso gana cutlass. Es el MISMO camino que corria antes:
        # un solo GEMM sobre el peso mergeado y despues SiLU.
        gu = ops.cutlass_scaled_mm(a_i8, w8.t(), sa.view(-1, 1),
                                   esc.view(1, -1), torch.float16)
        return torch.nn.functional.silu(gu[:, :n]) * gu[:, n:]

    return sk12_gateup_silu_gemm(a_i8, w8[:n], w8[n:], sa, esc[:n], esc[n:])


def _sk12_op_fake(a_i8: torch.Tensor, w8: torch.Tensor, esc: torch.Tensor,
                  sa: torch.Tensor) -> torch.Tensor:
    return torch.empty((a_i8.shape[0], w8.shape[0] // 2),
                       dtype=torch.float16, device=a_i8.device)


def registrar() -> bool:
    """Registra `vllm::genesis_sk12_gateup_silu`. Idempotente."""
    global _registrado
    if _registrado:
        return True
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="genesis_sk12_gateup_silu",
        op_func=_sk12_op,
        mutates_args=[],
        fake_impl=_sk12_op_fake,
    )
    _registrado = True
    return True


class _Stats:
    # `despachos` cuenta LLAMADAS al custom op, no despachos a SK-12: adentro
    # el op elige SK-12 o cutlass segun M. Para ver cual eligio, usar
    # GENESIS_PN119_DEBUG=N, que loguea la rama desde el propio op.
    despachos = 0
    salteados_sin_int8 = 0
    fallos = 0


STATS = _Stats()


try:
    _DEBUG_CADA = int(os.environ.get("GENESIS_PN119_DEBUG", "0"))
except ValueError:
    _DEBUG_CADA = 0


def gate_up_silu(gate_up_proj, x):
    """`SiLU(gate)*up` por SK-12, o None si no aplica (el caller sigue igual)."""
    if not activo():
        return None
    from vllm._genesis.int8_mlp_dispatch import ATTR

    estado = getattr(gate_up_proj, ATTR, None)
    if estado is None:
        STATS.salteados_sin_int8 += 1
        return None

    x2 = x.reshape(-1, x.shape[-1])

    # SIN gate por talla aca: lo decide el custom op en runtime. Ver _sk12_op.
    try:
        from vllm import _custom_ops as ops

        w8, esc = estado
        n = w8.shape[0] // 2
        a_i8, sa, _ = ops.scaled_int8_quant(x2, symmetric=True)
        out = torch.ops.vllm.genesis_sk12_gateup_silu(
            a_i8, w8, esc.float(), sa.reshape(-1).float())
        STATS.despachos += 1
        return out.reshape(*x.shape[:-1], n)
    except Exception as e:
        STATS.fallos += 1
        log.error("[PN119] SK-12 fallo, cae al camino normal: %s", e)
        return None


def stats() -> dict:
    return {"llamadas": STATS.despachos,
            "salteados_sin_int8": STATS.salteados_sin_int8,
            "fallos": STATS.fallos, "m_min": _m_min(),
            "invocaciones_op": _LLAMADAS[0]}


# El op se registra AL IMPORTAR. Llamar a direct_register_custom_op desde
# adentro del forward falla con "Attempted to call function marked as skipped":
# dynamo esta trazando y la registracion no es trazable.
if _ACTIVO:
    try:
        registrar()
    except Exception as _e:  # pragma: no cover
        log.error("[PN119] no se pudo registrar el custom op: %s", _e)

__all__ = ["gate_up_silu", "activo", "registrar", "stats", "STATS"]
