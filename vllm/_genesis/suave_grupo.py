# SPDX-License-Identifier: Apache-2.0
"""PN137 — suavizado por grupo plegado en los pesos: mas preciso que el dinamico y gratis.

La identidad que lo hace posible
--------------------------------
El checkpoint es GPTQ int4 con ``group_size=128`` y ``desc_act=False``, o sea::

    W[k][n] = (q[k][n] - z) * escala[k // 128][n]

Si el factor de suavizado ``s`` es CONSTANTE dentro de cada grupo de 128 canales de entrada, mover
la dificultad de la activacion al peso sale exacto y sin re-cuantizar nada::

    y = x W = (x / s) (s W)
    (s W)[k][n] = (q[k][n] - z) * escala[k // 128][n] * s_g      <- solo se multiplica la escala
    (x / s)_k                                                     <- se pliega en la RMSNorm previa

Las dos cosas se hacen al CARGAR. En ejecucion no queda nada: ni una pasada extra, ni un kernel mas
gordo. Y despues de suavizar, una sola escala estatica por capa alcanza y sobra.

Por que gana (no es un compromiso)
----------------------------------
Error relativo ``||q(x)-x||/||x||`` sobre activaciones reales del servidor:

| sitio | dinamico por token (lo de hoy) | suave grupo + escala estatica |
|---|---|---|
| qkv_proj | 0,1416 | **0,0296** |
| in_proj_qkvz | 0,0812 | **0,0337** |
| gate_up_proj | 0,0570 | **0,0275** |

O sea 2-4,8x MAS preciso, ademas de mas barato. El detalle esta en
``tests/proto/estaticos_offline.py``.

Donde se aplica y por que ahi
-----------------------------
Solo en los lineales alimentados por una RMSNorm, que son los unicos donde el factor de la
activacion se puede plegar: ``input_layernorm`` -> ``qkv_proj`` (capas de atencion) o
``in_proj_qkvz`` (capas GDN), y ``post_attention_layernorm`` -> ``gate_up_proj``. Son 128 de los
256 lineales. En ``down_proj`` el productor es SiLU x gate (no lineal, no se puede plegar) y en
``o_proj`` / ``out_proj`` es la salida de atencion, que no tiene norma previa. Ahi ademas el
suavizado por grupo seria PEOR que el dinamico, asi que la frontera cae sola en el lugar correcto.

Seguridad
---------
El pliegue son DOS mitades que tienen que aplicarse juntas: escalas del peso y peso de la norma. Si
se aplicara una sola, el modelo daria basura en silencio. Por eso se lleva la cuenta de las dos y
``verificar()`` corta el arranque si no coinciden.

Se prende con ``GENESIS_ENABLE_PN137_SUAVE=1`` y necesita el archivo de calibracion
(``GENESIS_SUAVE_ARCHIVO``), que sale de ``tests/bench/medicion/calibrar_suave.py``.
"""

from __future__ import annotations

import json
import logging
import os

import torch

log = logging.getLogger("genesis.pn137")

ACTIVO = os.environ.get("GENESIS_ENABLE_PN137_SUAVE", "0") == "1"
ARCHIVO = os.environ.get("GENESIS_SUAVE_ARCHIVO", "/escalas/suave.json")
# Etapa A: pliega los pesos pero deja la cuantizacion dinamica de siempre. Sirve para comprobar que
# el pliegue es exacto (la salida tiene que quedar igual). Etapa B agrega la escala estatica.
SOLO_PLIEGUE = os.environ.get("GENESIS_SUAVE_SOLO_PLIEGUE", "0") == "1"
G = 128

_cal: dict[str, dict] = {}
_escalas: dict[str, float] = {}     # escala estatica por capa, ya suavizada
_hecho_peso: set[str] = set()
_hecho_norma: set[str] = set()


def _cargar() -> None:
    if not ACTIVO:
        return
    try:
        with open(ARCHIVO) as f:
            crudo = json.load(f)
        for nombre, v in crudo.items():
            _cal[nombre] = v
            _escalas[nombre] = float(v["escala"])
            # En las capas GDN la MISMA norma alimenta tambien a in_proj_ba. Si se divide la norma
            # y no se compensa esa rama, el modelo da basura (medido: agujas 0/12). Hereda el mismo
            # factor, pero no escala estatica: sigue con la cuantizacion dinamica.
            if nombre.endswith("in_proj_qkvz"):
                _cal[nombre[: -len("in_proj_qkvz")] + "in_proj_ba"] = {"s": v["s"]}
        log.warning("PN137: %d capas calibradas de %s", len(_cal), ARCHIVO)
    except Exception as e:
        log.error("PN137: no se pudo leer %s (%s); queda inerte", ARCHIVO, e)


_cargar()


def activo() -> bool:
    return ACTIVO and bool(_cal)


def _clave(prefijo: str) -> str:
    """El calibrador nombra los archivos con los puntos cambiados por guiones bajos."""
    return prefijo.replace(".", "_")


def factor(prefijo: str, dispositivo, dtype=torch.float32):
    """Factor por grupo de esta capa, o None si no corresponde tocarla."""
    v = _cal.get(_clave(prefijo))
    if v is None:
        return None
    return torch.tensor(v["s"], device=dispositivo, dtype=dtype)


_actual = 0.0
# Cuenta de que camino tomo cada GEMM al TRAZARSE. Este parche ya quedo inerte dos veces (una por
# la cache de torch.compile y otra por el marcador del TextPatcher), asi que la senal es explicita.
_cuenta = [0, 0]   # [estatico, dinamico]


def marcar(prefijo: str) -> None:
    """Deja anotada la escala de la capa que esta por hacer el GEMM.

    Hace falta porque ``apply_gptq_marlin_linear`` no recibe el ``layer``, y agregarle un parametro
    obliga a tocar todas las firmas. Es seguro: entre ``marcar`` y el uso no hay nada asincronico
    (los dos corren en el mismo hilo del worker, uno detras del otro).
    """
    global _actual
    _actual = escala_de(prefijo)


def escala_actual() -> float:
    """Se CONSUME: si el GEMM que sigue no fue marcado, no hereda la escala del anterior."""
    global _actual
    e, _actual = _actual, 0.0
    # Solo se cuenta: un log aca adentro ROMPE el trace de fullgraph. Se lee desde afuera con
    # ``suave_grupo._cuenta``. Verificado 2026-09-17: 64 estaticos / 64 dinamicos por grafo, que son
    # los 128 sitios plegables sobre los 256 lineales.
    _cuenta[0 if e > 0.0 else 1] += 1
    return e


def escala_de(prefijo: str) -> float:
    """Escala estatica de activacion ya suavizada. 0 = seguir con la dinamica."""
    if SOLO_PLIEGUE:
        return 0.0
    return _escalas.get(_clave(prefijo), 0.0)


def cuantizar_estatico(x: torch.Tensor, escala: float):
    """int8 con escala FIJA: sin reduccion sobre la fila, que es todo el punto.

    A proposito NO es un ``custom_op``: aca no hay punteros ni mutacion, es aritmetica pura, asi que
    Dynamo la traza y **Inductor la fusiona** con lo que viene antes (la RMSNorm) en un unico kernel
    elementwise. Envuelta en un custom_op quedaba opaca y salian cinco kernels con un fp32 material
    en el medio, peor que el kernel fusionado de Marlin.
    """
    q = (x * (1.0 / escala)).round().clamp(-127, 127).to(torch.int8)
    e = torch.full((x.shape[0], 1), escala, dtype=torch.float32, device=x.device)
    return q, e


def aplicar_al_peso(layer, nombre_escalas: str) -> None:
    """Multiplica las escalas de grupo del GPTQ por el factor. Hay que llamarlo ANTES de que Marlin
    permute las escalas, o el indice de grupo ya no corresponde."""
    prefijo = getattr(layer, "prefix", None)
    if not prefijo or _clave(prefijo) in _hecho_peso:
        return
    par = getattr(layer, nombre_escalas, None)
    if par is None:
        return
    s = factor(prefijo, par.device, par.dtype)
    if s is None:
        return
    # vLLM guarda las escalas como [N, K/G] o [K/G, N] segun el camino; se detecta por cual
    # dimension coincide con la cantidad de grupos en vez de suponerlo.
    ng = s.numel()
    if par.dim() != 2 or ng not in par.shape:
        log.error("PN137: %s tiene escalas %s y el factor %d grupos; se saltea",
                  prefijo, tuple(par.shape), ng)
        return
    if par.shape[0] == ng and par.shape[1] != ng:
        par.data.mul_(s.unsqueeze(1))          # [K/G, N]
    elif par.shape[1] == ng and par.shape[0] != ng:
        par.data.mul_(s.unsqueeze(0))          # [N, K/G]
    else:
        log.error("PN137: %s tiene escalas %s y las dos dimensiones miden %d; ambiguo, se saltea",
                  prefijo, tuple(par.shape), ng)
        return
    _hecho_peso.add(_clave(prefijo))


def compensar_sin_cuantizar(lineal, prefijo: str) -> bool:
    """Compensa un lineal que NO pasa por Marlin multiplicando su peso fp16 por el factor.

    Hace falta para ``in_proj_ba`` de las capas GDN: es chico y el checkpoint lo deja sin
    cuantizar, asi que el gancho de las escalas de grupo nunca lo toca — pero lee la misma norma
    que ``in_proj_qkvz``, o sea que si no se lo compensa el modelo da basura.
    """
    if _clave(prefijo) in _hecho_peso:
        return True
    w = getattr(lineal, "weight", None)
    if w is None or w.dim() != 2 or not w.dtype.is_floating_point:
        return False                      # empaquetado o raro: no se toca
    s = factor(prefijo, w.device, torch.float32)
    if s is None:
        return False
    rep = s.repeat_interleave(G)
    if w.shape[1] != rep.numel():         # [salida, entrada]
        log.error("PN137: %s tiene peso %s y el factor cubre %d canales de entrada; se saltea",
                  prefijo, tuple(w.shape), rep.numel())
        return False
    w.data = (w.data.float() * rep.unsqueeze(0)).to(w.dtype)
    _hecho_peso.add(_clave(prefijo))
    return True


def _es_gemma(norma) -> bool:
    """¿La norma aplica x*(1+w) (estilo Gemma) o x*w (clasica)?

    Se mira el nombre de la clase Y se confirma con el valor tipico del peso: en la convencion de
    Gemma arranca en 0 y en la clasica en 1. Si las dos señales no coinciden, mejor abortar que
    plegar mal.
    """
    nombre = type(norma).__name__.lower()
    por_clase = "gemma" in nombre
    w = getattr(norma, "weight", None)
    if w is None:
        return por_clase
    cerca_de_cero = bool(w.data.float().abs().median() < 0.5)
    if por_clase != cerca_de_cero:
        raise RuntimeError(
            f"PN137: no se puede determinar la convencion de {type(norma).__name__}: la clase dice "
            f"{'(1+w)' if por_clase else 'w'} pero la mediana del peso es "
            f"{float(w.data.float().abs().median()):.3f}. Plegar con la convencion equivocada da "
            f"basura silenciosa.")
    return por_clase


def aplicar_a_la_norma(norma, prefijo_lineal: str) -> bool:
    """Divide el peso de la RMSNorm por el factor del lineal que alimenta."""
    if _clave(prefijo_lineal) in _hecho_norma:
        return True
    w = getattr(norma, "weight", None)
    if w is None:
        return False
    s = factor(prefijo_lineal, w.device, torch.float32)
    if s is None:
        return False
    rep = s.repeat_interleave(G)
    if rep.numel() != w.numel():
        log.error("PN137: la norma de %s tiene %d canales y el factor cubre %d; se saltea",
                  prefijo_lineal, w.numel(), rep.numel())
        return False
    # OJO CON LA CONVENCION. Este modelo usa GemmaRMSNorm, que aplica x * (1 + w) y por eso guarda
    # el peso inicializado en CERO. Dividir w/s seria incorrecto: lo que hay que escalar es (1+w).
    #     (1 + w') = (1 + w) / s   ->   w' = (1 + w)/s - 1
    # Con la convencion clasica (x * w, peso alrededor de 1) seria simplemente w/s. Confundirlas da
    # basura silenciosa: se midio agujas 0/12 y aceptacion 0,8%.
    if _es_gemma(norma):
        w.data = ((w.data.float() + 1.0) / rep - 1.0).to(w.dtype)
    else:
        w.data = (w.data.float() / rep).to(w.dtype)
    _hecho_norma.add(_clave(prefijo_lineal))
    return True


# Los lineales que leen la salida de cada norma. OJO: en GDN son DOS (in_proj_qkvz e in_proj_ba);
# compensar solo uno rompe el modelo en silencio.
CONSUMIDORES = {
    "input_layernorm": ("qkv_proj", "in_proj_qkvz", "in_proj_ba"),
    "post_attention_layernorm": ("gate_up_proj",),
}


def _consumidores(capa, nombre_norma):
    """Todos los lineales que consumen esa norma, y si quedo alguno sin reconocer."""
    if nombre_norma == "post_attention_layernorm":
        modulo = getattr(capa, "mlp", None)
    else:
        modulo = getattr(capa, "self_attn", None) or getattr(capa, "linear_attn", None)
    if modulo is None:
        return [], []
    conocidos, desconocidos = [], []
    for nombre, hijo in modulo.named_children():
        entrada = getattr(hijo, "input_size_per_partition", None) or getattr(hijo, "input_size", None)
        if entrada is None:
            continue                      # no es un lineal
        if nombre in CONSUMIDORES[nombre_norma]:
            conocidos.append(hijo)
        elif entrada == _ancho_norma(capa, nombre_norma):
            # otro lineal que tambien lee la salida de la norma y no esta en la lista
            desconocidos.append(nombre)
    return conocidos, desconocidos


def _ancho_norma(capa, nombre_norma):
    norma = getattr(capa, nombre_norma, None)
    w = getattr(norma, "weight", None) if norma is not None else None
    return w.numel() if w is not None else -1


def plegar_normas(model) -> int:
    """Segunda mitad del pliegue: recorre el modelo y divide cada RMSNorm por su factor."""
    if not activo():
        return 0
    n = 0
    for capa in _capas(model):
        for nombre_norma in CONSUMIDORES:
            norma = getattr(capa, nombre_norma, None)
            if norma is None:
                continue
            conocidos, desconocidos = _consumidores(capa, nombre_norma)
            if desconocidos:
                raise RuntimeError(
                    f"PN137: {nombre_norma} alimenta tambien a {desconocidos}, que no estan en la "
                    f"lista de consumidores. Dividir la norma sin compensarlos da basura "
                    f"silenciosa; hay que agregarlos a CONSUMIDORES o apagar PN137.")
            # los que no pasaron por Marlin (in_proj_ba) se compensan aca, sobre el peso fp16
            for lin in conocidos:
                pre = getattr(lin, "prefix", "")
                if pre and _clave(pre) not in _hecho_peso:
                    compensar_sin_cuantizar(lin, pre)
            # se pliega la norma solo si TODOS sus consumidores quedaron compensados
            prefijos = [getattr(l, "prefix", "") for l in conocidos]
            if not prefijos or not all(_clave(p) in _hecho_peso for p in prefijos if p):
                continue
            if aplicar_a_la_norma(norma, prefijos[0]):
                n += 1
    log.warning("PN137: %d normas plegadas (%d pesos ya plegados)", n, len(_hecho_peso))
    verificar()
    return n


def _capas(model):
    vistos, pila = set(), [model]
    while pila:
        m = pila.pop(0)
        if id(m) in vistos:
            continue
        vistos.add(id(m))
        capas = getattr(m, "layers", None)
        if capas is not None and len(capas) and any(hasattr(c, "mlp") for c in capas):
            return list(capas)
        for _, hijo in m.named_children():
            pila.append(hijo)
    return []


def verificar() -> None:
    """Las dos mitades tienen que coincidir. Si no, el modelo daria basura EN SILENCIO."""
    # Solo se comparan los consumidores PRINCIPALES (los que tienen escala estatica). Los
    # secundarios, como in_proj_ba, se compensan pero no son la clave de ninguna norma.
    principales = _hecho_peso & set(_escalas)
    solo_peso = principales - _hecho_norma
    solo_norma = _hecho_norma - principales
    if solo_peso or solo_norma:
        raise RuntimeError(
            f"PN137: el pliegue quedo a medias ({len(solo_peso)} con el peso escalado y sin norma, "
            f"{len(solo_norma)} al reves). Aplicar una sola mitad da basura silenciosa. "
            f"Ejemplos: {sorted(solo_peso)[:3] or sorted(solo_norma)[:3]}")
    log.warning("PN137: pliegue completo y consistente en %d normas (%d pesos, con secundarios)%s",
                len(_hecho_norma), len(_hecho_peso),
                " [solo pliegue, cuantizacion dinamica]" if SOLO_PLIEGUE else "")
