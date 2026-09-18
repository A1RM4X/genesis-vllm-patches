# SPDX-License-Identifier: Apache-2.0
"""PN131 sin cirugia de texto: el backend de atencion como SUBCLASE registrada.

El camino viejo reescribe cinco pedazos de ``v1/attention/backends/triton_attn.py``. Funciona,
pero cuando upstream mueve una linea el ancla no aparece, ``TextPatcher`` devuelve SKIPPED y el
servidor arranca **sin el parche y sin gritar**: la atencion cae al kernel generico y lo unico
que se nota es que anda mas lento. Eso es lo caro de cada migracion.

vLLM v0.29.0 expone un registro de backends (``v1/attention/backends/registry.register_backend``)
que acepta un camino de clase como string, resuelto tarde. Con eso PN131 pasa a ser lo que
siempre fue conceptualmente — un backend de atencion — en vez de un conjunto de parches de texto:

    register_backend(AttentionBackendEnum.TRITON_ATTN,
                     "vllm._genesis.sk18_backend.BackendSK18")

y se prende con ``--attention-backend TRITON_ATTN``, que es una bandera de vLLM de verdad.

Heredar tambien puede fallar callado — sobreescribir un metodo que el padre ya no define no da
error, simplemente no reemplaza nada. Por eso ``_exige()`` verifica al importar que cada metodo
que venimos a reemplazar siga existiendo en el padre, y si no, revienta con un mensaje que dice
cual. Ruidoso a proposito: es justamente lo que el camino de anclas no hacia.

Que reemplaza, y por que (es lo mismo que hacian los cinco sub-parches):

* ``BuilderSK18.get_cudagraph_support``: el decode entero se lanza con ``cuLaunchKernel``, que se
  graba bien, pero solo para el decode UNIFORME (1 + tokens de MTP). De ahi UNIFORM_BATCH.
* ``BuilderSK18.build``: deja ``query_start_loc_cpu`` en la metadata. Se necesita en CPU y ya
  esta calculado, asi que tomarlo de aca evita sincronizar contra la GPU.
* ``ImplSK18.forward``: el decode por SK-18h. Los pasos con prefill caen al padre.
* ``ImplSK18.do_kv_cache_update``: la escritura del KV con el layout propio (K por token, V por
  dimension, escalas int16) dentro de la misma reserva de 520 B por token-cabeza.
"""

from __future__ import annotations

import logging

from vllm.v1.attention.backend import AttentionCGSupport, AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadataBuilder,
)

from vllm._genesis import sk18_attn as _g131

log = logging.getLogger("genesis.pn131")


def _exige(cls: type, *nombres: str) -> None:
    """Falla al importar si el padre ya no define lo que venimos a reemplazar.

    Sin esto, una subclase que sobreescribe un metodo renombrado por upstream se carga sin
    ruido y no reemplaza nada — el mismo fallo silencioso que tenia el camino de anclas, solo
    que disfrazado de herencia.
    """
    faltan = [n for n in nombres if not any(n in vars(c) for c in cls.__mro__)]
    if faltan:
        raise RuntimeError(
            f"[PN131] {cls.__name__} ya no define {faltan}: vLLM cambio el backend de "
            "atencion y hay que revisar sk18_backend.py antes de seguir."
        )


_exige(TritonAttentionImpl, "forward", "do_kv_cache_update")
_exige(TritonAttentionMetadataBuilder, "build")


class BuilderSK18(TritonAttentionMetadataBuilder):
    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        if _g131.dtype_activo(getattr(vllm_config.cache_config, "cache_dtype", "")):
            # El decode entero va en PTX por cuLaunchKernel y se graba, pero solo el decode
            # uniforme (1 + tokens de MTP) tiene forma constante.
            return AttentionCGSupport.UNIFORM_BATCH
        padre = getattr(super(), "get_cudagraph_support", None)
        if padre is not None:
            return padre(vllm_config, kv_cache_spec)
        return cls._cudagraph_support

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        m = super().build(common_prefix_len, common_attn_metadata, fast_build)
        # Ya esta calculado y en CPU: tomarlo de aca ahorra una sincronizacion con la GPU.
        m.genesis_qsl_cpu = common_attn_metadata.query_start_loc_cpu
        return m


class ImplSK18(TritonAttentionImpl):
    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        # `attn_metadata` viene en None en la pasada de perfilado. El padre lo maneja arriba de
        # todo y sale temprano; el parche de texto entraba DESPUES de eso, asi que nunca lo veia.
        # Al heredar interceptamos antes, y sin esta guarda el arranque muere con
        # "AttributeError: 'NoneType' object has no attribute 'use_cascade'".
        if attn_metadata is not None and _g131.activo(self, layer):
            assert attn_metadata.use_cascade is False, \
                "[PN131] el decode entero no contempla atencion en cascada"
            return _g131.forward(self, layer, query, kv_cache, attn_metadata, output)
        return super().forward(layer, query, key, value, kv_cache, attn_metadata, output,
                               output_scale, output_block_scale)

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        if _g131.activo(self, layer):
            if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
                # La atencion de encoder usa Q/K/V directos, sin cache. Mismo orden que el
                # padre: primero la salida temprana, despues nuestra escritura.
                return None
            return _g131.escribir(self, layer, key, value, kv_cache, slot_mapping)
        return super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)


class BackendSK18(TritonAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN"

    @staticmethod
    def get_impl_cls() -> type[TritonAttentionImpl]:
        return ImplSK18

    @staticmethod
    def get_builder_cls() -> type[TritonAttentionMetadataBuilder]:
        return BuilderSK18


def registrar() -> str:
    """Registra el backend por camino de clase. Devuelve el camino, para el log."""
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    camino = f"{BackendSK18.__module__}.{BackendSK18.__qualname__}"
    register_backend(AttentionBackendEnum.TRITON_ATTN, camino)
    log.info("[PN131] TRITON_ATTN registrado como %s", camino)
    return camino
