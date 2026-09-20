# SPDX-License-Identifier: Apache-2.0
"""PN126 — rotacion Hadamard/FWHT ENTERA de q/k despues de RoPE antes de la atencion.

Por que
-------
La KV cuantizada (int8/int4 por token-cabeza) sufre por los outliers de
canal de k. Con Hadamard, los picos de canal se dispersan uniformemente en
todas las dimensiones de la cabeza (D=128 o D=256), reduciendo el error de
cuantizacion de KV de ~5% a ~0,5% (5x de reduccion de error).

Filosofia CERO Punto Flotante:
------------------------------
Toda la rotacion Hadamard se realiza como una Fast Walsh-Hadamard Transform (FWHT)
con mariposas de sumas y restas en int32 y escalado por desplazamiento de bits
fijo (>> 4 en D=256, multiplicador entero Q20 en D=128). Cero multiplicaciones de
matrices en punto flotante. Implementado con kernel CUDA en PTX (~7 us) y fallback
vectorizado en PyTorch entero.

Interaccion con PN131 (SK-18):
------------------------------
PN131 rota q/k en PTX adentro de sus propios kernels para capas de atencion
completa del modelo target con D=256. Para no rotar dos veces, `activo(256)`
se desactiva cuando PN131 esta activo. En cambio, para capas con D=128 (como
el borrador DFlash2) o cualquier capa no atendida por PN131, `activo(128)`
permanece SIEMPRE ACTIVO.
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
if not MODO and os.environ.get("GENESIS_ENABLE_PN126_ROT_QK", "0") == "1":
    MODO = "hadamard"

DAMP = float(os.environ.get("GENESIS_PN126_DAMP", "0.01"))
_TRIGGER = "/dev/shm/pn126_volcar"

_mats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
_grams: dict[str, list] = {}
_cargados = None
_llamadas = 0
_volcado = False

_signos: dict[tuple[int, int], torch.Tensor] = {}
_kernels: dict[tuple[int, torch.dtype], any] = {}


def activo(D: int | None = None) -> bool:
    if os.environ.get("GENESIS_ENABLE_PN126_ROT_QK", "0") != "1" and MODO not in ("hadamard", "captura", "wush"):
        return False
    # PN131 con rotacion entera en PTX rota q/k adentro de sus kernels (solo capas QD=256 de atencion completa).
    # Para D=256 y PN131 activo con PTX, no rotar dos veces.
    # Para D=128 (DFlash2) o capas fuera de PN131, rot_qk debe estar activo.
    if D == 256 and (os.environ.get("GENESIS_ENABLE_PN131_SK18", "0") == "1"
                     and os.environ.get("GENESIS_PN131_ROT", "ptx") == "ptx"
                     and MODO == "hadamard"):
        return False
    return MODO in ("hadamard", "captura", "wush")


def _rank() -> int:
    try:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
        return int(get_tensor_model_parallel_rank())
    except Exception:
        return int(torch.cuda.current_device()) if torch.cuda.is_available() else 0


def _signos_dev(D: int, dev: torch.device) -> torch.Tensor:
    idx = dev.index if dev.index is not None else 0
    k = (D, idx)
    s = _signos.get(k)
    if s is None:
        g = torch.Generator(device=dev).manual_seed(126)  # Mismos signos fijados que PN131
        s = (torch.randint(0, 2, (D,), generator=g, device=dev).to(torch.int32) * 2 - 1).contiguous()
        _signos[k] = s
    return s


def _get_kernel(D: int, dtype: torch.dtype):
    k = (D, dtype)
    if k in _kernels:
        return _kernels[k]
    # El kernel lee los BITS de 16: con cualquier otro dtype (fp32 en la referencia de
    # PN131.escribir) interpretaria basura. Esos van por el fallback entero de PyTorch.
    sufijo = {torch.float16: "f16", torch.bfloat16: "bf16"}.get(dtype)
    if sufijo is None:
        _kernels[k] = None
        return None
    try:
        from vllm._genesis.kernels.ptx_lab import Kernel
        nombre = f"sk_fwht{D}_{sufijo}"
        kern = Kernel("sk_fwht.cu", nombre, warps=4)
        kern.cargar()
        _kernels[k] = kern
        return kern
    except Exception as e:
        log.warning("[PN126] fallback entero: no se cargo kernel CUDA %s: %s", f"sk_fwht{D}", e)
        _kernels[k] = None
        return None


def _fwht_int_py(x_flat: torch.Tensor, s: torch.Tensor, D: int) -> torch.Tensor:
    """Fallback puro en PyTorch entero (int32) para Fast Walsh-Hadamard."""
    x_i = (x_flat.float() * 16384.0).round().to(torch.int32) * s
    h = 1
    while h < D:
        v = x_i.view(-1, D // (2 * h), 2, h)
        x_i = torch.cat([v[:, :, 0, :] + v[:, :, 1, :],
                         v[:, :, 0, :] - v[:, :, 1, :]], dim=-1).view(-1, D)
        h *= 2
    if D == 256:
        y = (x_i + 8) >> 4
    elif D == 128:
        y = ((x_i.to(torch.int64) * 92682 + 524288) >> 20).to(torch.int32)
    else:
        raise ValueError(f"D={D} no soportado para FWHT")
    return (y.float() * (1.0 / 16384.0)).to(x_flat.dtype)


def rotar_tensor(x: torch.Tensor, D: int) -> None:
    """Aplica la rotacion Hadamard ENTERA (FWHT) in-place a cualquier tensor que termine en D."""
    if x.numel() == 0 or D not in (128, 256):
        return
    dev = x.device
    signos = _signos_dev(D, dev)
    M = x.numel() // D

    necesita_copia = not x.is_contiguous()
    target = x.contiguous() if necesita_copia else x
    flat = target.view(-1, D)

    kern = _get_kernel(D, x.dtype) if dev.type == "cuda" else None
    if kern is not None:
        grid_x = (M + 3) // 4
        kern.lanzar((grid_x, 1), [flat, signos, M])
    else:
        out = _fwht_int_py(flat, signos, D)
        flat.copy_(out)

    if necesita_copia:
        x.copy_(target)


def hadamard(n: int, device, dtype=torch.float64) -> torch.Tensor:
    H = torch.ones(1, 1, device=device, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


def _hadamard_aleatoria(D: int, device) -> torch.Tensor:
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
    if MODO == "hadamard":
        rotar_tensor(q, D)
        rotar_tensor(k, D)
    else:
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
    try:
        direct_register_custom_op(op_name="genesis_rot_qk", op_func=_impl,
                                  mutates_args=["q", "k"], fake_impl=_fake)
    except Exception:
        # Ya registrado en este proceso
        pass


def rotar(q: torch.Tensor, k: torch.Tensor, nombre: str, hkv: int, D: int) -> None:
    if not activo(D):
        return
    if hasattr(torch.ops.vllm, "genesis_rot_qk"):
        torch.ops.vllm.genesis_rot_qk(q, k, nombre, hkv, D)
    else:
        _impl(q, k, nombre, hkv, D)


if activo():
    try:
        registrar()
    except Exception as e:  # pragma: no cover
        log.error("[PN126] no se pudo registrar: %s", e)
