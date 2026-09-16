# SPDX-License-Identifier: Apache-2.0
"""WUSH (arXiv 2512.00956) para el MLP W4A4 simulado de PN118 — diagnostico.

Reemplaza la Hadamard fija por bloques de 64 por una transformacion por bloque
calibrada con los momentos de segundo orden de pesos (W) y activaciones (X):

    W' W'^T = damp(W_b^T W_b / R)     X' X'^T = damp(X_b^T X_b / n)
    W'^T X' = U S V^T
    activacion:  x~ = T_x x,   T_x = H S^-1/2 U^T W'^T
    peso (fila): w~ = T_w w,   T_w = H S^-1/2 V^T X'^T        T_w^T T_x = I

La asignacion es CRUZADA (la T de la activacion se arma con el momento del
peso): medido en q.k reales, la asignacion literal duplica el error.

Dos modos, por GENESIS_PN118_WUSH:
  "captura:<dir>"  acumula X_b^T X_b por capa del MLP en el camino INT8 normal
                   (necesita --enforce-eager) y vuelca <dir>/rank{r}.pt cuando
                   aparece /dev/shm/wush_volcar.
  "aplicar:<dir>"  en la carga arma T_x/T_w por capa en la GPU y el forward
                   simulado (GENESIS_PN118_FAKE) las usa en lugar de la rotacion.
"""

from __future__ import annotations

import logging
import math
import os

import torch

log = logging.getLogger("genesis.wush")

B = 64
_CRUDO = os.environ.get("GENESIS_PN118_WUSH", "").strip()
MODO, _, DIR = _CRUDO.partition(":")
MODO = MODO.lower()
DAMP = float(os.environ.get("GENESIS_PN118_WUSH_DAMP", "0.01"))
_TRIGGER = "/dev/shm/wush_volcar"

_grams: dict[str, list] = {}
_llamadas = 0
_volcado = False
_cargados = None


def capturando() -> bool:
    return MODO == "captura"


def aplicando() -> bool:
    return MODO == "aplicar"


def _rank() -> int:
    from vllm._genesis.int8_mlp_dispatch import _rank_tp
    return _rank_tp()


def _bloques(t: torch.Tensor) -> torch.Tensor:
    K = t.shape[-1]
    assert K % B == 0, f"K={K} no es multiplo de {B}"
    return t.reshape(*t.shape[:-1], K // B, B)


@torch.no_grad()
def acumular(nombre: str, x2: torch.Tensor) -> None:
    global _llamadas, _volcado
    xb = _bloques(x2.float())                                   # [N, nb, B]
    g = torch.einsum("nbi,nbj->bij", xb, xb).double()
    ent = _grams.get(nombre)
    if ent is None:
        _grams[nombre] = [g, x2.shape[0]]
    else:
        ent[0] += g
        ent[1] += x2.shape[0]
    _llamadas += 1
    if _llamadas % 256 == 0 and not _volcado and os.path.exists(_TRIGGER):
        os.makedirs(DIR, exist_ok=True)
        dest = os.path.join(DIR, f"rank{_rank()}.pt")
        torch.save({k: (v[0].cpu(), v[1]) for k, v in _grams.items()}, dest)
        _volcado = True
        log.warning("[WUSH] volcadas %d capas (%d tokens en la primera) -> %s",
                     len(_grams), next(iter(_grams.values()))[1], dest)


def _hadamard(n: int, device) -> torch.Tensor:
    H = torch.ones(1, 1, device=device, dtype=torch.float64)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


def _damp(M: torch.Tensor) -> torch.Tensor:
    d = M.diagonal(dim1=-2, dim2=-1).mean(-1)
    eye = torch.eye(M.shape[-1], device=M.device, dtype=M.dtype)
    return M + DAMP * d.clamp_min(1e-12)[:, None, None] * eye


@torch.no_grad()
def preparar(layer, nombre: str, w8: torch.Tensor, esc: torch.Tensor) -> None:
    """Arma T_x, T_w [nb, B, B] float32 en la GPU y las cuelga de la capa."""
    global _cargados
    if not aplicando() or ".mlp." not in nombre:
        return
    if _cargados is None:
        _cargados = torch.load(os.path.join(DIR, f"rank{_rank()}.pt"), map_location="cpu")
        log.warning("[WUSH] cargados %d gramianos (rank %d)", len(_cargados), _rank())
    ent = _cargados.get(nombre)
    if ent is None:
        log.warning("[WUSH] sin gramiano para %s: queda sin transformar", nombre)
        return
    dev = w8.device
    Mx = ent[0].to(dev) / ent[1]                                  # [nb, B, B]
    nb = Mx.shape[0]
    Mw = torch.zeros(nb, B, B, device=dev, dtype=torch.float64)
    for lo in range(0, w8.shape[0], 1024):
        W = _bloques(w8[lo:lo + 1024].double() * esc[lo:lo + 1024].double().view(-1, 1))
        Mw += torch.einsum("rbi,rbj->bij", W, W)
    Mw /= w8.shape[0]
    Wp = torch.linalg.cholesky(_damp(Mw))
    Xp = torch.linalg.cholesky(_damp(Mx))
    U, S, Vh = torch.linalg.svd(Wp.transpose(1, 2) @ Xp)
    H = _hadamard(B, dev)
    Sm = S.rsqrt()[:, :, None]
    Tx = H @ (Sm * (U.transpose(1, 2) @ Wp.transpose(1, 2)))
    Tw = H @ (Sm * (Vh @ Xp.transpose(1, 2)))
    err = (Tw.transpose(1, 2) @ Tx - torch.eye(B, device=dev, dtype=torch.float64)).abs().max().item()
    # fp16: 128 capas en float32 son ~450 MiB por placa y el arranque muere con
    # OOM al cargar el drafter; los temporales float64 se devuelven al allocator.
    layer._g118_wush = (Tx.half().contiguous(), Tw.half().contiguous())
    del Mx, Mw, Wp, Xp, U, S, Vh, Tx, Tw
    torch.cuda.empty_cache()
    if err > 1e-6:
        log.warning("[WUSH] %s: T_w^T T_x != I (max %.2e)", nombre, err)


def transformar(t: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """t [..., nb*B] -> por bloque T @ t_b."""
    tb = _bloques(t)
    return torch.einsum("bij,...bj->...bi", T.to(t.dtype), tb).reshape(t.shape)
