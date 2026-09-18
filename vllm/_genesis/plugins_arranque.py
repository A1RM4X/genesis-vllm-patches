# SPDX-License-Identifier: Apache-2.0
"""El punto por donde Genesis se registra DENTRO de cada proceso de vLLM.

Por que hace falta
------------------
``apply_all`` corre como un proceso aparte — el entrypoint hace
``python3 -m vllm._genesis.patches.apply_all`` y despues ``exec vllm serve`` — asi que todo lo
que haga en MEMORIA se pierde al cruzar el exec. Los parches de texto sobreviven porque editan
archivos; un ``register_backend`` no.

Eso no es un detalle: se midio. Con PN131 registrado desde apply_all, el arranque decia
"applied ... TRITON_ATTN registrado" y el servidor levantaba y generaba bien, pero SK-18 nunca
corria — el decode caia al kernel generico y lo unico que se notaba era la velocidad. El aviso
de una sola vez que hay en ``sk18_attn.forward`` es lo que lo delato.

``vllm.plugins.load_general_plugins()`` se llama en TODOS los procesos (servidor y workers), una
sola vez por proceso, y su cuerpo es identico en v0.27.1 y en v0.29.0. Por eso el enganche es
una sola ancla ahi, y todo lo demas se resuelve por herencia y registros nativos.

Que se registra aca
-------------------
* PN131 con ``GENESIS_PN131_NATIVO=1``: el backend de atencion SK-18 por ``register_backend``.

Es el lugar para lo que venga: una ``QuantizationConfig`` propia para el camino W4A8, capas
por ``PluggableLayer.register_oot``, etc. Todo eso tambien necesita correr en el proceso que
sirve, no en el que parchea.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.plugins")

_cargado = False


def _prendido(nombre: str, defecto: str = "0") -> bool:
    return os.environ.get(nombre, defecto).strip().lower() in ("1", "true", "yes", "on")


def cargar() -> None:
    """Idempotente y a prueba de balas: no puede tumbar un proceso de vLLM.

    Un fallo aca se registra como ERROR con el detalle, pero NO se propaga: esta funcion la
    llama vLLM desde su propio arranque y romperla dejaria el servidor sin levantar por algo
    que puede ser opcional. El ruido queda en el log, que es donde se mira cuando algo anda
    mas lento de lo que deberia.
    """
    global _cargado
    if _cargado:
        return
    _cargado = True

    _avisar_si_falta_disco()

    if _prendido("GENESIS_DIAG_LMHEAD"):
        try:
            from vllm._genesis.diag_lm_head import enganchar

            enganchar()
        except Exception as e:                                   # noqa: BLE001
            log.error("[DIAG] no se pudo enganchar la firma de lm_head (%s: %s)",
                      type(e).__name__, e)

    if _prendido("GENESIS_ENABLE_PN131_SK18") and _prendido("GENESIS_PN131_NATIVO"):
        try:
            from vllm._genesis.sk18_backend import registrar

            registrar()
        except Exception as e:                                   # noqa: BLE001
            log.error("[PN131] no se pudo registrar el backend SK-18 (%s: %s). La atencion "
                      "va a correr por el kernel generico.", type(e).__name__, e)


def _avisar_si_falta_disco(minimo_gib: float = 10.0) -> None:
    """Grita si el disco de las caches JIT esta casi lleno.

    No es paranoia de manual: los unicos arranques que dejaron el modelo roto en v0.29.0
    ocurrieron con el disco raiz al 96-100% y 8,18 GiB de RAM disponible para leer un
    checkpoint de 18,12 GiB. Despues de liberar 31 GB, 22 arranques seguidos salieron sanos.
    No esta probado que sea la causa — ver la nota del proyecto — pero un disco sin lugar
    mientras se escriben las caches de torch.compile, Inductor y Triton es una fuente de
    artefactos truncados que despues se cargan en silencio, y eso explicaria que la falla
    fuera por arranque y desapareciera sola al haber espacio.

    Avisar cuesta nada. Perseguir un fantasma otra vez, bastante.
    """
    try:
        import shutil

        libre = shutil.disk_usage("/root/.cache").free / 2 ** 30
        if libre < minimo_gib:
            log.warning("[Genesis] quedan %.1f GiB en el disco de las caches JIT (menos de "
                        "%.0f). Las caches de torch.compile/Inductor/Triton se escriben ahi "
                        "durante el arranque; sin lugar pueden quedar truncadas y cargarse "
                        "despues sin avisar.", libre, minimo_gib)
    except Exception:
        pass    # un chequeo de cortesia nunca puede molestar al arranque
