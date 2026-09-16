# SPDX-License-Identifier: Apache-2.0
"""PN125 — runtime: escalas positivas para Marlin W4A8-INT8.

El bug
------
Con ``VLLM_MARLIN_INPUT_DTYPE=int8``, ``marlin_act_int8_process_scales`` pasa
las escalas por grupo a int16 relativas a ``s.max()`` y el kernel las toma sin
signo. Los checkpoints AutoRound meten el signo en la escala (``(q-8)*s`` con
``s < 0``): en noon-at-cgn/Qwen3.8-27B-Uncensored-W4A16-AutoRound el 50,5% de
las escalas son negativas en TODAS las proyecciones. Resultado medido con una
capa real (tests/proto/marlin_w4a8_capa.py): W4A16 0,03% de error, W4A8
3.754% — el modelo responde "!!!!" en todo. Normalizar por ``|s|.max()``
dejando el signo en el int16 da 5.183%: el kernel exige escalas >= 0.

El arreglo
----------
En los grupos con ``s < 0``: ``s -> |s|`` y ``q -> 16 - q``, que es exacto
salvo ``q = 0`` (-8 -> +8, no representable en [-8, 7]). Casi todos los grupos
negativos usan el nivel -8, asi que recortar a +7 cuesta ~2,6% de error en el
peso; buscando la escala positiva de minimo error por grupo baja a ~2,5%
(``GENESIS_PN125_BUSQUEDA=1``, por defecto). Solo corre con activaciones int8
y pesos uint4b8; W4A16 no se toca.
"""

from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("genesis.pn125")

_BUSQUEDA = os.environ.get("GENESIS_PN125_BUSQUEDA", "1").strip() not in ("0", "false", "no", "off")
_FACTORES = (6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5)
_FILAS = 1024


class _Stats:
    capas = 0
    grupos = 0
    negativos = 0
    err2 = 0.0
    ref2 = 0.0


STATS = _Stats()


def _corregir_bloque(q: torch.Tensor, s: torch.Tensor, G: int) -> tuple[torch.Tensor, torch.Tensor]:
    """q [n, g*G] uint4 (int16), s [n, g] -> (q', s') con s' >= 0."""
    n, g = s.shape
    qg = q.view(n, g, G)
    sg = s.unsqueeze(-1)
    neg = sg < 0
    qa = torch.where(neg, (16 - qg).clamp(max=15), qg)
    sa = sg.abs()
    if _BUSQUEDA:
        W = (qg - 8).float() * sg.float()
        mejor_e = ((qa - 8).float() * sa.float() - W).square().sum(-1, keepdim=True)
        base = W.abs().amax(-1, keepdim=True).clamp_min(1e-12)
        for f in _FACTORES:
            sc = base / f
            qb = (W / sc).round().clamp(-8, 7)
            e = (qb * sc - W).square().sum(-1, keepdim=True)
            m = neg & (e < mejor_e)
            qa = torch.where(m, (qb + 8).to(qa.dtype), qa)
            sa = torch.where(m, sc.to(sa.dtype), sa)
            mejor_e = torch.where(m, e, mejor_e)
    return qa.view(n, g * G), sa.squeeze(-1)


@torch.no_grad()
def positivizar(layer, w_q_name: str, w_s_name: str, c) -> None:
    from vllm.model_executor.parameter import permute_param_layout_
    from vllm.scalar_type import scalar_types

    if c.act_type != torch.int8 or c.weight_type != scalar_types.uint4b8 or c.has_g_idx:
        return
    # Con PN130 (kernel propio con escalas int16 con signo) no hace falta tocar
    # los pesos: el arreglo es exacto en el kernel.
    try:
        from vllm._genesis import marlin_s16
        if marlin_s16.ACTIVO_Y_CARGADO:
            return
    except Exception:
        pass
    pq = getattr(layer, w_q_name, None)
    ps = getattr(layer, w_s_name, None)
    if pq is None or ps is None:
        return
    permute_param_layout_(pq, input_dim=0, output_dim=1, packed_dim=0)   # [K/8, N]
    permute_param_layout_(ps, input_dim=0, output_dim=1)                 # [K/G, N]
    if not bool((ps.data < 0).any()):
        return
    qw = pq.data.t().contiguous()            # [N, K/8] int32, nibble j en bits 4j
    s = ps.data.t().contiguous()             # [N, g]
    N, P = qw.shape
    g = s.shape[1]
    K8 = P * 8
    G = c.group_size if c.group_size > 0 else K8
    sh = torch.arange(0, 32, 4, device=qw.device, dtype=torch.int32)
    q_out = torch.empty_like(qw)
    s_out = torch.empty_like(s)
    for lo in range(0, N, _FILAS):
        hi = min(N, lo + _FILAS)
        q = ((qw[lo:hi].unsqueeze(-1) >> sh) & 0xF).reshape(hi - lo, K8).to(torch.int16)
        util = g * G
        q2, s2 = _corregir_bloque(q[:, :util], s[lo:hi], G)
        if util < K8:
            q2 = torch.cat([q2, q[:, util:]], dim=1)
        W = (q[:, :util].view(hi - lo, g, G) - 8).float() * s[lo:hi].float().unsqueeze(-1)
        W2 = (q2[:, :util].view(hi - lo, g, G) - 8).float() * s2.float().unsqueeze(-1)
        STATS.err2 += float((W2 - W).square().sum())
        STATS.ref2 += float(W.square().sum())
        packed = (q2.to(torch.int64).view(hi - lo, P, 8) << sh.to(torch.int64)).sum(-1)
        q_out[lo:hi] = torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32)
        s_out[lo:hi] = s2
    STATS.capas += 1
    STATS.grupos += s.numel()
    STATS.negativos += int((s < 0).sum())
    pq.data = q_out.t().contiguous()
    ps.data = s_out.t().contiguous()
    if STATS.capas in (1, 50, 100, 200, 300, 400):
        log.warning("[PN125] %d capas corregidas: %.1f%% de escalas negativas, error de peso %.3f%%",
                    STATS.capas, 100 * STATS.negativos / max(1, STATS.grupos),
                    100 * (STATS.err2 / max(STATS.ref2, 1e-30)) ** 0.5)
