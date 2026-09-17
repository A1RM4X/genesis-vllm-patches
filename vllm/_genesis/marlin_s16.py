# SPDX-License-Identifier: Apache-2.0
"""PN130 — Marlin W4A8-INT8 propio con escalas de grupo int16 CON signo.

El bug (vllm#48905): en el camino de activaciones int8, Marlin guarda las
escalas por grupo en int16 relativas a ``s.max()`` y el kernel las lee con
``reinterpret_cast<uint16_t*>`` (``marlin_template.h``). AutoRound deja ~50% de
escalas negativas -> basura. PN125 lo esquivaba cambiando los pesos (q->16-q,
s->|s|), a costa de ~2,6% de error en el peso por el nivel -8 que no tiene par.

Aca se arregla en el kernel: el mismo Marlin de vLLM v0.27.1
(``kernels/marlin_s16/src``, instanciacion solo s8 x uint4b8 x fp16, grupos 128 y
por canal) con ``int16_t``, compilado como extension propia y registrado como
``genesis_marlin::marlin_gemm_s16``. Las escalas se normalizan por ``|s|.max()``
para que las negativas entren en el int16. Medido en k_proj real de noon: error
0,87% (el de cuantizar x a int8) contra 2,82% con PN125 y 0,03% en W4A16.

La extension se compila una vez (~1 min, nvcc del contenedor) en
``GENESIS_MARLIN_S16_BUILD`` (default ``/root/.cache/vllm/genesis_marlin_s16``,
montado por modelo) desde ``apply_all``; los workers solo la cargan.
"""

from __future__ import annotations

import glob
import logging
import os
import subprocess
import sys

import torch

log = logging.getLogger("genesis.pn130")

_AQUI = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_AQUI, "kernels", "marlin_s16")
BUILD = os.environ.get("GENESIS_MARLIN_S16_BUILD", "/root/.cache/vllm/genesis_marlin_s16")
_cargado = False
# La firma del op cambio cuando PN140 sumo `a_sums_or_none` antes de `workspace`. Una .so vieja
# en la cache y un fuente nuevo (o al reves) dejaban la llamada con un argumento corrido, y el
# error era "Expected a value of type 'Tensor' for argument 'workspace' but instead found int".
# Se resuelve mirando el schema al cargar, asi que las dos firmas andan.



def activo() -> bool:
    return os.environ.get("GENESIS_ENABLE_PN130_MARLIN_S16", "0").strip().lower() in ("1", "true", "yes", "on")


def _so() -> str | None:
    c = glob.glob(os.path.join(BUILD, "genesis_marlin_s16*.so"))
    return c[0] if c else None


def _huella() -> str:
    """Hash de las fuentes del kernel. Si cambia, la .so de la cache ya no sirve."""
    import hashlib
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(_SRC, "src", "*")) +
                    glob.glob(os.path.join(_SRC, "inc", "*")) +
                    [os.path.join(_SRC, "torch_bindings.cpp")]):
        if os.path.isfile(f):
            h.update(os.path.basename(f).encode())
            with open(f, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()[:16]


def construir() -> str:
    """Compila la extension si no esta, o si el fuente cambio. Devuelve la ruta de la .so.

    El chequeo de huella no es un lujo: una .so vieja en la cache con un fuente nuevo deja la
    llamada con un argumento corrido (a PN140 le sumo `a_sums_or_none` antes de `workspace`), y
    eso tira el arranque con "expected at most 19 argument(s) but received 20" — o, peor, lo
    dejaba pasar en silencio por un camino distinto.
    """
    so = _so()
    marca = os.path.join(BUILD, "huella.txt")
    if so:
        try:
            with open(marca) as f:
                vieja = f.read().strip()
        except OSError:
            vieja = ""
        if vieja == _huella():
            return so
        log.warning("[PN130] el fuente cambio (%s -> %s): se recompila", vieja or "sin marca",
                    _huella())
        for viejo in glob.glob(os.path.join(BUILD, "*.so")):
            os.remove(viejo)
        so = None
    if so:
        return so
    env = dict(os.environ, GENESIS_MARLIN_S16_BUILD=BUILD, TORCH_CUDA_ARCH_LIST="8.6")
    r = subprocess.run([sys.executable, os.path.join(_SRC, "build.py")], env=env,
                       capture_output=True, text=True)
    so = _so()
    if r.returncode != 0 or not so:
        raise RuntimeError("compilacion de genesis_marlin_s16 fallo:\n" + (r.stdout + r.stderr)[-3000:])
    try:
        with open(os.path.join(BUILD, "huella.txt"), "w") as f:
            f.write(_huella())
    except OSError:
        pass
    return so


def cargar() -> bool:
    """Carga la .so y registra el fake para torch.compile. Idempotente."""
    global _cargado
    if _cargado:
        return True
    so = _so()
    if so is None:
        log.info("[PN130] extension todavia sin compilar en %s (la compila apply_all)", BUILD)
        return False
    torch.ops.load_library(so)
    from torch.library import register_fake

    log.info("[PN130] .so cargada desde %s", so)

    @register_fake("genesis_marlin::marlin_gemm_s16")
    def _fake(a, c, b_q_weight, b_bias, b_scales, a_scales, global_scale, b_zeros, g_idx, perm,
              *resto):
        # `resto` es (a_sums?, workspace, b_type_id, size_m, size_n, size_k, ...): el a_sums
        # puede o no estar, asi que el indice se corre. Lo unico que necesita el fake es la
        # forma de la salida.
        size_m, size_n = resto[3], resto[4]   # (a_sums, workspace, b_type_id, m, n)
        dtype = a.dtype
        if dtype not in (torch.half, torch.bfloat16):
            dtype = b_scales.dtype
        return torch.empty((size_m, size_n), device=a.device, dtype=dtype)

    _cargado = True
    return True


def procesar_escalas(s: torch.Tensor):
    """Como ``marlin_act_int8_process_scales`` pero por ``|s|.max()``: conserva el signo."""
    m = s.abs().max()
    factor = 1 / 4096 * m.float()
    s = s / m * 4096
    s = s.round().to(torch.int16).view(s.dtype)
    return s, factor


def marlin_gemm(a, c, b_q_weight, b_bias, b_scales, a_scales, global_scale, b_zeros, g_idx,
                perm, workspace, b_q_type, size_m, size_n, size_k, is_k_full=True,
                use_atomic_add=False, use_fp32_reduce=False, is_zp_float=False, a_sums=None):
    """Misma firma que ``vllm._custom_ops.marlin_gemm``, mas el ``a_sums`` de PN140.

    SIN condicionales: esto corre adentro de una region trazada por Dynamo, que hornea como
    constante cualquier decision que se tome aca (y ni siquiera sabe iterar `_schema.arguments`,
    falla con "missing tp_iter"). Que la .so coincida con el fuente lo garantiza `construir()`,
    que la recompila cuando cambia el hash de las fuentes.
    """
    return torch.ops.genesis_marlin.marlin_gemm_s16(
        a, c, b_q_weight, b_bias, b_scales, a_scales, global_scale, b_zeros, g_idx, perm,
        a_sums, workspace, b_q_type.id, size_m, size_n, size_k, is_k_full, use_atomic_add,
        use_fp32_reduce, is_zp_float)


ACTIVO_Y_CARGADO = False
if activo():
    try:
        ACTIVO_Y_CARGADO = cargar()
    except Exception as e:  # pragma: no cover
        log.error("[PN130] no se pudo cargar: %s", e)
