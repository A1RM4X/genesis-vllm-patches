# SPDX-License-Identifier: Apache-2.0
"""PN135 — residual + RMSNorm + cuantizacion int8 en un solo kernel (SK-20).

Hoy cada sitio de norma dispara tres kernels y la activacion se recorre tres veces:

    h, r  = fused_add_rms_norm(h, r, w)     # inductor, 3,3 us por sitio
    h_q,s = per_token_quant_int8(h)         # 2,2 us
    s     = s * input_global_scale          # 1,3 us, para multiplicar M floats

``sk20_norm_quant`` hace las tres cosas de una. La idea que lo abarata: **la normalizacion se
cancela en los enteros**. Con y = r * inv_rms * w y q = round(y * 127 / max|y|), el inv_rms esta
arriba y abajo, asi que q = round(r*w*127/max|r*w|) y el inv_rms solo hace falta UNA vez por
fila, para la escala fp32. No hay que normalizar 5120 valores: alcanza con un escalar.

Por eso el kernel NO escribe la activacion normalizada en fp16: escribe el residuo actualizado y
directamente el int8 con su escala. El fp16 que devuelve la norma es un buffer sin escribir que
solo sirve de acarreo: el int8 y la escala viajan en un diccionario indexado por ``data_ptr`` y
el gancho de Marlin (PN134/PN135 en ``marlin_utils.py``) los levanta en vez de cuantizar.

Eso obliga a un cuidado: el consumidor de la norma TIENE que ser el lineal Marlin W4A8 que
levanta el int8. Por eso el enganche se hace capa por capa (``envolver(model)``), atando cada
norma a su lineal consumidor y solo si ese lineal es Marlin con factor global. Si algo no cierra,
la norma corre por el camino de siempre.

Medido con grafos CUDA (M=5/K=5120, RTX 3090 @220 W): 3 kernels 6,0 us -> SK-20 5,76 us; con
M>=512 (prefill) 642 -> 528 us (1,22x).

La suma del residuo usa ``add.f16x2`` del hardware. Es la UNICA excepcion a la regla de trabajar
en enteros, y esta medida: el sumador entero equivalente esta en el mismo .cu (``-DENTSUMA=1``),
da exactamente los mismos bits (verificado elemento a elemento) y cuesta 2,6 us mas por kernel,
el 44% del total. Ahi el entero no compra ni precision ni determinismo.

SIN CABLEAR, Y POR QUE
---------------------
El traspaso del int8 por ``data_ptr`` (el diccionario de abajo) **no sobrevive a torch.compile**:
la rama del gancho de Marlin se evalua al TRAZAR, cuando el tensor es un FakeTensor sin puntero,
asi que quedaria horneada la rama "cuantizar normal" y el lineal leeria el buffer de acarreo, que
nunca se escribe. Es la misma trampa de PN134 (ver la memoria del proyecto), pero aca el sintoma
seria basura silenciosa en vez de un parche inerte.

La forma correcta es no pasar nada por Python: buffers int8 y de escalas FIJOS y compartidos,
colgados de la capa consumidora antes de trazar, que el kernel muta (``mutates_args``) y que el
gancho lee por identidad de objeto (``layer``, que si existe en ``GPTQMarlinLinearMethod.apply``).
Es mas invasivo, y el premio medido punta a punta es 0,4% en decode y 0,7% en prefill, asi que
queda escrito y sin cablear hasta que haya una razon mas fuerte.

El kernel en si esta medido y verificado (tests/proto/sk20_norm_quant_test.py,
sk20_grafo.py, sk20_costo_entero.py). ``envolver()`` queda listo para el dia que se haga.
"""

from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("genesis.pn135")

HILOS = int(os.environ.get("GENESIS_PN135_HILOS", 512))
_k = {}
_stash: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
_avisado = False


def activo() -> bool:
    return os.environ.get("GENESIS_ENABLE_PN135_NORM_QUANT", "0") == "1"


def _kernel():
    dev = torch.cuda.current_device()
    if dev not in _k:
        from vllm._genesis.kernels.ptx_lab import Kernel
        x = Kernel("sk20_norm_quant.cu", "sk20_norm_quant", defs=[f"-DHILOS={HILOS}"],
                   warps=HILOS // 32)
        x.cargar()
        _k[dev] = x
    return _k[dev]


@torch.library.custom_op("genesis::pn135_norm_quant", mutates_args=())
def _pn135(x: torch.Tensor, res: torch.Tensor, w: torch.Tensor,
           g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Devuelve (acarreo fp16 SIN escribir, residuo nuevo, int8, escala fp32).

    Op propia para que torch.compile no la abra y quede grabada tal cual en el CUDA graph.
    """
    global _avisado
    m, k = x.shape
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    esc = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    res_out = torch.empty_like(x)
    gf = g if g.dtype == torch.float32 else g.float()
    _kernel().lanzar((m, 1), [x.view(torch.int16), res.view(torch.int16), w.view(torch.int16),
                              gf.reshape(-1).view(torch.int32), res_out.view(torch.int16), q,
                              esc.view(torch.int32), k, x.stride(0), 1], 4 * k)
    # acarreo: mismo dtype y forma que la activacion normalizada, pero nadie lo lee (el consumidor
    # levanta el int8 del diccionario). No se escribe: esa es justamente la pasada que se ahorra.
    acarreo = torch.empty_like(x)
    _stash[acarreo.data_ptr()] = (q, esc)
    if not _avisado:
        _avisado = True
        log.warning("PN135: residual + RMSNorm + int8 en un solo kernel (SK-20)")
    return acarreo, res_out, q, esc


@_pn135.register_fake
def _pn135_fake(x: torch.Tensor, res: torch.Tensor, w: torch.Tensor, g: torch.Tensor):
    m, k = x.shape
    return (torch.empty_like(x), torch.empty_like(x),
            torch.empty((m, k), dtype=torch.int8, device=x.device),
            torch.empty((m, 1), dtype=torch.float32, device=x.device))


def tomar(x: torch.Tensor):
    """El gancho de Marlin pide aca el int8 ya cuantizado para este tensor de acarreo."""
    if not _stash:
        return None
    par = _stash.pop(x.data_ptr(), None)
    if len(_stash) > 512:                       # si algun sitio no lo levanta, no se acumula
        _stash.clear()
    return par


def _factor_global(lineal) -> torch.Tensor | None:
    """El factor global de la activacion del lineal, si es un Marlin W4A8 que lo usa."""
    for nombre in ("input_global_scale", "input_scale"):
        g = getattr(lineal, nombre, None)
        if isinstance(g, torch.Tensor) and g.numel() == 1:
            return g
    qm = getattr(lineal, "quant_method", None)
    g = getattr(qm, "input_global_scale", None)
    return g if isinstance(g, torch.Tensor) and g.numel() == 1 else None


def _consumidor(capa):
    """El primer lineal que lee la salida de cada norma de la capa."""
    fuera = {}
    at = getattr(capa, "self_attn", None) or getattr(capa, "linear_attn", None)
    for nombre in ("qkv_proj", "in_proj_qkvz", "in_proj"):
        lin = getattr(at, nombre, None) if at is not None else None
        if lin is not None:
            fuera["input_layernorm"] = lin
            break
    mlp = getattr(capa, "mlp", None)
    lin = getattr(mlp, "gate_up_proj", None) if mlp is not None else None
    if lin is not None:
        fuera["post_attention_layernorm"] = lin
    return fuera


def envolver(model) -> int:
    """Ata cada RMSNorm de capa a su lineal consumidor. Devuelve cuantas quedaron fusionadas."""
    if not activo():
        return 0
    n = 0
    capas = getattr(getattr(model, "model", model), "layers", None) or []
    for capa in capas:
        for nombre, lineal in _consumidor(capa).items():
            norma = getattr(capa, nombre, None)
            w = getattr(norma, "weight", None)
            if not isinstance(w, torch.Tensor) or w.dtype != torch.float16:
                continue
            _envolver_una(norma, lineal)
            n += 1
    if n:
        log.warning("PN135: %d normas fusionadas con su cuantizacion", n)
    return n


def _envolver_una(norma, lineal):
    """El factor global se resuelve en cada llamada (Python, o sea al trazar/capturar): cuando se
    envuelve todavia no paso ``process_weights_after_loading`` y el tensor puede cambiar."""
    original = norma.forward

    def forward(x, residual=None, *a, **kw):
        g = _factor_global(lineal)
        if (g is None or residual is None or x.dim() != 2 or x.dtype != torch.float16
                or (x.shape[1] & 7) or x.stride(1) != 1):
            return original(x, residual, *a, **kw)
        acarreo, res_out, _q, _e = torch.ops.genesis.pn135_norm_quant(x, residual,
                                                                     norma.weight, g)
        return acarreo, res_out

    norma.forward = forward
