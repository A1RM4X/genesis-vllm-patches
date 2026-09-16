# SPDX-License-Identifier: Apache-2.0
"""PN126 — rotacion de q/k despues de RoPE (Hadamard o WUSH) antes de la atencion.

Por que
-------
La KV cuantizada (fp8, int8/int4 por token-cabeza) sufre por los outliers de
canal de k. Con los q/k reales de este modelo (volcados PN123, capas 3/35) el
error de la salida de atencion fue:

    fp8 sin rotar        1,82 / 2,40 %
    int8 sin rotar       (peor que con Hadamard)
    int8 + Hadamard      0,37 / 0,55 %
    int4 + Hadamard      6,0  / 8,6  %

Como ``q~ . k~ = q . k`` exacto, la rotacion no necesita un kernel de atencion
nuevo: se aplica a q y k antes de ``self.attn`` y la KV guarda k ya rotada. v no
se toca.

Modos (``GENESIS_PN126_ROT``)
-----------------------------
``hadamard``         R = H_256 * diag(signos), la misma para q y k (ortogonal).
``captura:<dir>``    acumula por capa y cabeza KV ``sum q q^T`` y ``sum k k^T``
                     y los vuelca a ``<dir>/rank{r}.pt`` cuando aparece
                     ``/dev/shm/pn126_volcar``. Correr con ``--enforce-eager``.
``wush:<dir>``       T por cabeza KV con los gramianos capturados (arXiv
                     2512.00956), asignacion CRUZADA medida en q/k reales:
                         Q'Q'^T = damp(M_q), K'K'^T = damp(M_k), Q'^T K' = U S V^T
                         T_q = H S^-1/2 V^T K'^T      T_k = H S^-1/2 U^T Q'^T
                     con T_q^T T_k = I.

Va como custom op in-place (``mutates_args=["q", "k"]``): el primer llamado
ocurre en el profile run, antes de capturar CUDA graphs, y ahi se construyen
las matrices en la GPU.
"""

from __future__ import annotations

import logging
import math
import os

import torch

log = logging.getLogger("genesis.pn126")

_CRUDO = os.environ.get("GENESIS_PN126_ROT", "").strip()
MODO, _, DIR = _CRUDO.partition(":")
MODO = MODO.lower()
DAMP = float(os.environ.get("GENESIS_PN126_DAMP", "0.01"))
_TRIGGER = "/dev/shm/pn126_volcar"

_mats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
_grams: dict[str, list] = {}
_cargados = None
_llamadas = 0
_volcado = False


def activo() -> bool:
    # PN131 con rotacion entera en PTX rota q/k adentro de sus kernels: no rotar dos veces.
    if (os.environ.get("GENESIS_ENABLE_PN131_SK18", "0") == "1"
            and os.environ.get("GENESIS_PN131_ROT", "ptx") == "ptx" and MODO == "hadamard"):
        return False
    return MODO in ("hadamard", "captura", "wush")


def _rank() -> int:
    try:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
        return int(get_tensor_model_parallel_rank())
    except Exception:
        return int(torch.cuda.current_device())


def hadamard(n: int, device, dtype=torch.float64) -> torch.Tensor:
    H = torch.ones(1, 1, device=device, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


def _hadamard_aleatoria(D: int, device) -> torch.Tensor:
    # Signos fijos (semilla fija, generados en la GPU): la matriz tiene que ser la
    # misma en todos los ranks y en todos los arranques, porque la KV y la cache
    # de prefijos guardan k ya rotada.
    g = torch.Generator(device=device).manual_seed(126)
    s = torch.randint(0, 2, (D,), generator=g, device=device).to(torch.float64) * 2 - 1
    return hadamard(D, device) * s[None, :]


def _damp(M: torch.Tensor) -> torch.Tensor:
    d = M.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-12)
    return M + DAMP * d[..., None, None] * torch.eye(M.shape[-1], device=M.device, dtype=M.dtype)


def construir_wush(Mq: torch.Tensor, Mk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mq, Mk [H, D, D] float64 -> (T_q, T_k) [H, D, D] con T_q^T T_k = I."""
    D = Mq.shape[-1]
    Qp = torch.linalg.cholesky(_damp(Mq))
    Kp = torch.linalg.cholesky(_damp(Mk))
    U, S, Vh = torch.linalg.svd(Qp.transpose(-1, -2) @ Kp)
    H = hadamard(D, Mq.device)
    Sm = S.rsqrt()[..., :, None]
    Tq = H @ (Sm * (Vh @ Kp.transpose(-1, -2)))
    Tk = H @ (Sm * (U.transpose(-1, -2) @ Qp.transpose(-1, -2)))
    return Tq, Tk


def _matrices(nombre: str, hkv: int, D: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    global _cargados
    m = _mats.get(nombre)
    if m is not None:
        return m
    if MODO == "hadamard":
        R = _hadamard_aleatoria(D, device).expand(hkv, D, D)
        m = (R.half().contiguous(), R.half().contiguous())
    else:
        if _cargados is None:
            _cargados = torch.load(os.path.join(DIR, f"rank{_rank()}.pt"), map_location="cpu")
            log.warning("[PN126] gramianos cargados: %d capas (rank %d)", len(_cargados), _rank())
        ent = _cargados.get(nombre)
        if ent is None:
            log.warning("[PN126] %s sin gramianos: queda sin rotar", nombre)
            eye = torch.eye(D, device=device, dtype=torch.float16).expand(hkv, D, D).contiguous()
            m = (eye, eye)
        else:
            Mq = ent["q"].to(device, torch.float64) / ent["nq"]
            Mk = ent["k"].to(device, torch.float64) / ent["nk"]
            Tq, Tk = construir_wush(Mq, Mk)
            err = (Tq.transpose(-1, -2) @ Tk - torch.eye(D, device=device, dtype=torch.float64)).abs().max().item()
            if err > 1e-5:
                log.warning("[PN126] %s: T_q^T T_k != I (max %.2e)", nombre, err)
            m = (Tq.half().contiguous(), Tk.half().contiguous())
    _mats[nombre] = m
    return m


def _acumular(nombre: str, q: torch.Tensor, k: torch.Tensor, hkv: int, D: int) -> None:
    global _llamadas, _volcado
    T = q.shape[0]
    qh = q.reshape(T, hkv, -1, D).float()
    kh = k.reshape(T, hkv, D).float()
    gq = torch.einsum("tgid,tgie->gde", qh, qh).double()
    gk = torch.einsum("tgd,tge->gde", kh, kh).double()
    e = _grams.get(nombre)
    nq = T * qh.shape[2]
    if e is None:
        _grams[nombre] = [gq, gk, nq, T]
    else:
        e[0] += gq; e[1] += gk; e[2] += nq; e[3] += T
    _llamadas += 1
    if _llamadas % 64 == 0 and not _volcado and os.path.exists(_TRIGGER):
        os.makedirs(DIR, exist_ok=True)
        dest = os.path.join(DIR, f"rank{_rank()}.pt")
        torch.save({n: {"q": v[0].cpu(), "k": v[1].cpu(), "nq": v[2], "nk": v[3]} for n, v in _grams.items()}, dest)
        _volcado = True
        log.warning("[PN126] volcados %d capas -> %s", len(_grams), dest)


def _impl(q: torch.Tensor, k: torch.Tensor, nombre: str, hkv: int, D: int) -> None:
    if MODO == "captura":
        _acumular(nombre, q, k, hkv, D)
        return
    Mq, Mk = _matrices(nombre, hkv, D, q.device)
    T = q.shape[0]
    qh = q.reshape(T, hkv, -1, D)
    kh = k.reshape(T, hkv, D)
    q.copy_(torch.einsum("gde,tgie->tgid", Mq.to(q.dtype), qh).reshape(q.shape))
    k.copy_(torch.einsum("gde,tge->tgd", Mk.to(k.dtype), kh).reshape(k.shape))


def _fake(q, k, nombre, hkv, D) -> None:
    return None


def registrar() -> None:
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(op_name="genesis_rot_qk", op_func=_impl,
                              mutates_args=["q", "k"], fake_impl=_fake)


def rotar(q: torch.Tensor, k: torch.Tensor, nombre: str, hkv: int, D: int) -> None:
    torch.ops.vllm.genesis_rot_qk(q, k, nombre, hkv, D)


if activo():
    try:
        registrar()
    except Exception as e:  # pragma: no cover
        log.error("[PN126] no se pudo registrar: %s", e)
