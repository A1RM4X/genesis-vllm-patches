# SPDX-License-Identifier: Apache-2.0
"""PN144 — bajar el residual del borrador DFlash2 para que entre en fp16. Exacto, sin costo.

El problema
-----------
El borrador DFlash2 corre su residual ~100 veces mas caliente que un modelo normal. Medido
con la traza de ``diag_dflash`` (una pasada, modulo por modulo, en fp16):

    11  layers.0.self_attn.o_proj    norma 6.29e+04  max  9.496
    17  layers.0.mlp.down_proj       norma 1.98e+04  max  4.256
    19  layers.0  (salida)           norma 1.76e+05  max 50.080   <-- fp16 topea en 65.504
    20  layers.1.input_layernorm     NaN entra=2

Despues de UNA capa el residual ya esta al 76% del techo de fp16; la suma residual de la capa
siguiente lo cruza, da inf, y el NaN se come el resto del borrador. En bf16 no pasa porque su
exponente llega a 3e38 — por eso ``--dtype bfloat16`` lo "arregla". Pero ese flag es GLOBAL y
se lleva puesto al modelo grande: nos saca de fp16 y con eso perdemos PN130 (Marlin W4A8, los
~2600 de prefill), las activaciones int8 y el KV int8/int4 de PN131. Medido: -13% de decode y
+40% de TTFT a 50k. Se pagaba una fortuna por el rango del exponente de cinco capitas.

El arreglo
----------
La RMSNorm es invariante a escala: ``rms_norm(x/c) == rms_norm(x)``. Entonces, si se divide
por ``c`` TODO lo que entra al residual, cada norma ve exactamente lo mismo que antes y la
salida del borrador es la misma — pero el residual corre ``c`` veces mas frio y no topea.

Lo que entra al residual es, y esto es todo:

* la semilla: la salida de ``embed_input_ids`` (el embedding compartido del target);
* por capa, ``self_attn.o_proj`` (unica salida de la atencion al residual);
* por capa, ``mlp.down_proj`` (unica salida del MLP al residual).

``fc`` queda AFUERA a proposito, aunque salga igual de caliente (norma 6.27e+04): no entra al
residual, su pico es 6.736 y no desborda, y su ``eps`` esta COMPARTIDO con el de ``k_norm``
(``_rms_norm_eps`` sale de ``attn0.q_norm``), asi que escalarlo obligaria a corregir un epsilon
que tambien usa un camino sin escalar. No hacia falta.

Lo que NO hay que tocar, justamente porque su entrada ya viene normalizada y no cambia:
``qkv_proj``, ``gate_up_proj``, los ``q_norm``/``k_norm``, y las dos ``*_conv.kernel_projection``
(la traza confirma que las dos reciben la salida de un layernorm).

El epsilon, que es donde esto se rompe si uno no mira
----------------------------------------------------
La RMSNorm no divide por ``rms(x)`` sino por ``sqrt(mean(x^2) + eps)``, y ``eps`` NO escala.
Mientras ``mean(x^2)`` sea mucho mayor que ``eps`` da igual, pero en la capa 0 el residual es
solo la semilla del embedding, que es chiquita:

    mean(x^2) original = 3,2e-05  vs eps 1e-06  -> eps pesa 3%
    mean(x^2) / 32^2   = 3,2e-08  vs eps 1e-06  -> eps pesa 97%   <-- la capa 0 sale cualquier cosa

Medido: sin corregir esto la aceptacion queda en 3,81 en vez de 5,26. Por eso hay que dividir
tambien ``variance_epsilon`` por ``c^2`` en las normas que ven el residual escalado — las dos
de cada capa y la final. Las ``q_norm``/``k_norm`` NO se tocan: ven valores sin escalar.

Por que es gratis de verdad
---------------------------
No es un kernel ni un cast en el lazo: son once tensores de escalas que se dividen UNA vez al
cargar, antes del repack de Marlin. Y ``c`` es potencia de dos, asi que dividir un fp16 es
bit-exacto (baja el exponente y nada mas): cero error introducido. El margen esta medido — la
escala mas chica de las once es 1,822e-02, que sobre 32 da 5,69e-04, nueve veces por encima
del minimo normal de fp16 (6,104e-05).

La unica operacion en caliente es dividir la semilla del embedding, que es un elementwise
sobre [tokens, 5120] una vez por paso del borrador.

Es el mismo argumento de invariancia computacional que ya esta anotado como tarea del
proyecto (rotacion del residual / SliceGPT), pero en su forma mas barata: un escalar.

Como verificarlo
----------------
No hay que creerle a la matematica: la transformacion es neutra o no lo es, y se ve en la
aceptacion. La referencia esta medida en bf16 — accept-len 5,26 a 1k y 4,89 a 50k. Si con
esto en fp16 da lo mismo, es correcta; si da distinto, hay un camino al residual que no esta
en la lista de arriba y hay que buscarlo con la traza.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.dflash2_escala")

#: Los unicos dos modulos por capa que escriben en el residual.
_SUFIJOS = (".self_attn.o_proj", ".mlp.down_proj")

#: Normas que ven el residual ESCALADO, y a las que por lo tanto hay que bajarles el epsilon.
#: `q_norm`/`k_norm` quedan afuera: su entrada no esta escalada.
_NORMAS_POR_CAPA = ("input_layernorm", "post_attention_layernorm")


def divisor() -> float:
    """0 (apagado) o el divisor. `GENESIS_PN144_ESCALA_RESIDUAL=1` significa el valor por
    defecto de 32, que es el que tiene el margen medido."""
    crudo = os.environ.get("GENESIS_PN144_ESCALA_RESIDUAL", "0").strip()
    if crudo in ("", "0", "off", "no", "false"):
        return 0.0
    if crudo in ("1", "on", "yes", "true"):
        return 32.0
    try:
        return float(crudo)
    except ValueError:
        log.error("[PN144] GENESIS_PN144_ESCALA_RESIDUAL=%r no es un numero; apagado", crudo)
        return 0.0


def _escalar_modulo(mod, c: float, nombre: str) -> bool:
    """Divide por `c` la escala de dequantizacion de un lineal. True si hizo algo."""
    for attr in ("weight_scale", "scales"):
        t = getattr(mod, attr, None)
        if t is not None and hasattr(t, "data"):
            t.data.div_(c)
            return True
    # Un borrador SIN cuantizar (o una capa que quedo densa) tiene el peso a secas.
    w = getattr(mod, "weight", None)
    if w is not None and hasattr(w, "data") and w.data.is_floating_point():
        w.data.div_(c)
        return True
    log.warning("[PN144] %s no tiene ni weight_scale ni weight denso: no se escalo", nombre)
    return False


def _bajar_epsilon(norma, c: float) -> bool:
    """``eps`` no escala con la entrada: para que la norma siga siendo invariante hay que
    dividirlo por ``c^2`` (va sumado a ``mean(x^2)``, que escala cuadraticamente)."""
    if norma is None:
        return False
    eps = getattr(norma, "variance_epsilon", None)
    if eps is None:
        return False
    norma.variance_epsilon = float(eps) / (c * c)
    return True


def enganchar() -> None:
    """Envuelve ``load_weights`` y ``forward`` del borrador. No puede tumbar el arranque."""
    c = divisor()
    if not c:
        return

    from vllm.model_executor.models import qwen3_dflash

    modelo_cls = qwen3_dflash.DFlashQwen3Model
    cabeza_cls = qwen3_dflash.DFlashQwen3ForCausalLM

    # --- 1) las escalas, una vez, despues de cargar y ANTES del repack de Marlin ---
    load_original = cabeza_cls.load_weights

    def load_weights(self, weights):
        salida = load_original(self, weights)
        m = self.model
        tocados, normas = [], []

        for i, capa in enumerate(m.layers):
            for suf in _SUFIJOS:
                obj = capa
                for pieza in suf.strip(".").split("."):
                    obj = getattr(obj, pieza, None)
                    if obj is None:
                        break
                if obj is not None and _escalar_modulo(obj, c, f"layers.{i}{suf}"):
                    tocados.append(f"layers.{i}{suf}")
            for nn in _NORMAS_POR_CAPA:
                if _bajar_epsilon(getattr(capa, nn, None), c):
                    normas.append(f"layers.{i}.{nn}")
        if _bajar_epsilon(getattr(m, "norm", None), c):
            normas.append("norm")

        if not tocados:
            log.error("[PN144] NO se escalo ni un tensor: el borrador va a desbordar en fp16 "
                      "y aceptar 0. Cambiaron los nombres de los modulos?")
        log.warning("[PN144] residual /%g en %d pesos y epsilon /%g en %d normas. La salida "
                    "no cambia: la norma es invariante a escala, y el epsilon se corrige "
                    "para que lo siga siendo.", c, len(tocados), c * c, len(normas))
        return salida

    cabeza_cls.load_weights = load_weights

    # --- 2) la semilla del residual: el embedding compartido del target ---
    # Sin esto la transformacion NO es neutra: el embedding entra al residual sin pasar por
    # ninguna norma propia, asi que si todo lo demas baja y el no, su peso relativo sube c
    # veces. Es chico (norma 3,6 contra 6,3e+04 de o_proj) pero no es cero.
    fwd_original = modelo_cls.forward

    def forward(self, input_ids, positions, input_embeds=None):
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)
        return fwd_original(self, input_ids, positions, input_embeds / c)

    modelo_cls.forward = forward
    log.warning("[PN144] enganchado: escalas al cargar y semilla del embedding en cada paso "
                "(divisor %g)", c)
