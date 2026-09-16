# SPDX-License-Identifier: Apache-2.0
"""PN118 — cuantiza el MLP de AWQ int4 a INT8 per-canal en el ARRANQUE.

Que problema resuelve
---------------------
El bloque `gate_up + SiLU` del MLP es el 43% del chunk de prefill y hoy corre
en fp16 por Marlin. Medido en esta maquina con la GPU libre, misma forma
(M=2048 K=5120 N=8704x2):

    Marlin W4A16 + SiLU     6,330 ms    57,7 TOPS   <- lo que corre hoy
    cutlass INT8 + SiLU     3,591 ms   101,7 TOPS   1,76x
    SK-12 W8A8 fusionado    3,775 ms    96,7 TOPS   1,68x

Las dos opciones rapidas piden pesos INT8 per-canal, que este checkpoint no
trae. PN118 los produce en la carga, una vez.

Es un TEXT PATCH, no un monkeypatch
-----------------------------------
vLLM carga el modelo en procesos WORKER separados y `apply_all` no corre ahi
(verificado: cero lineas de Genesis con prefijo `Worker_TP` en el log). Un
monkeypatch aplicado en el padre nunca llega al worker, asi que no puede tocar
la carga de pesos. La primera version de PN118 era un monkeypatch: se
instalaba, el log lo confirmaba, y no convertia una sola capa.

Que hace, en orden
------------------
1. En `process_weights_after_loading`: convierte los tres tensores AWQ a INT8
   per-canal y los deja en `layer._genesis_int8_mlp`.
2. Libera `weight_packed`, `weight_scale` y `weight_zero_point`, y NO llama al
   camino original, asi que Marlin nunca repaqueta nada.
3. En `apply_weights`: reemplaza el forward. Cuantiza la activacion per-token a
   INT8 y despacha por `cutlass_scaled_mm`.

El paso 3 es el que acelera. Sin el, el parche convertiria los pesos y el
forward seguiria yendo por Marlin sin usarlos.

Donde engancha, y por que ahi
-----------------------------
En `CompressedTensorsWNA16.process_weights_after_loading`, que es el unico
punto donde los tres tensores de AWQ (`weight_packed`, `weight_scale`,
`weight_zero_point`) estan juntos y todavia sin repaquetar a Marlin.

Ademas cae ANTES del profiling de memoria de vLLM. Eso importa: vLLM
dimensiona el KV cache midiendo la VRAM libre despues de cargar los pesos, asi
que si la conversion ya ocurrio, el KV se achica solo y no hay que tocar
--gpu-memory-utilization a mano.

La contabilidad de VRAM
-----------------------
Medido del checkpoint:

    MLP hoy (int4 + escalas + zp)   9,277 GiB  ->  4,639 por GPU
    MLP en int8 per-canal          15,938 GiB  ->  7,969 por GPU
                                    delta al REEMPLAZAR: +3,330 GiB

Con 1,14 GiB libres hay que sacarle 2,19 GiB al KV: 701.376 -> ~566.000 tokens
de contexto, contra un working set de 380.000. Entra.

Ese delta es la DIFERENCIA, no la suma: el int4 se libera apenas se convierte.
NO hay fallback a Marlin, a proposito: conservar los originales haria que el
modelo pague las dos copias y el parche no serviria para nada. Si el camino
INT8 falla en una capa, el arranque falla — y es lo que se quiere, porque una
capa cayendo a Marlin en silencio seria peor que un error.

`GENESIS_PN118_KEEP_INT4=1` conserva los originales, pero es solo para depurar
una capa puntual; en ese modo el parche no ahorra VRAM.

Precision
---------
Pasar de 4 bits por grupos de 128 a 8 bits per-canal solo funciona si las
escalas de grupo de un canal no varian mucho. Medido en este checkpoint el
rango es 1,7-2,0x y el error extra queda en ~1%, sobre el 3-5% que el int4 ya
arrastra. Con `GENESIS_PN118_AUDIT=1` se loguea el error real por capa al
convertir, para verificarlo en vez de confiar.

Alcance
-------
Solo capas `.mlp.`, que son el 70,2% de los FLOPs de GEMM. GDN y atencion
quedan intactas: son el 29,8% restante y su conversion sumaria VRAM sin el
mismo retorno.

Nota: si se cambia a un checkpoint que YA venga en W8A8 INT8 (por ejemplo
orcarouter/Qwen3.8-27B-Uncensored-INT8), este parche sobra: vLLM despacha solo
por `CompressedTensorsW8A8Int8` y nunca llega a `CompressedTensorsWNA16`.
PN118 se salta sin hacer nada.
"""

from __future__ import annotations

import logging

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.pn118_int8_mlp_boot_quant")

GENESIS_PN118_MARKER = "[Genesis PN118: MLP AWQ int4 -> INT8 per-canal]"

ANCHOR_OLD = (
    "    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:\n"
    "        self.kernel.process_weights_after_loading(layer)\n"
    "\n"
    "    def apply_weights(\n"
    "        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None\n"
    "    ) -> torch.Tensor:\n"
    "        return self.kernel.apply_weights(layer, x, bias)\n"
)

ANCHOR_NEW = (
    "    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:\n"
    "        # " + GENESIS_PN118_MARKER + "\n"
    "        # Si la capa quedo en INT8, sus tensores AWQ ya fueron liberados:\n"
    "        # llamar al kernel original explotaria. Por eso se corta aca.\n"
    "        from vllm._genesis import int8_mlp_dispatch as _g118\n"
    "        if _g118.convertir_capa(layer):\n"
    "            return\n"
    "        self.kernel.process_weights_after_loading(layer)\n"
    "\n"
    "    def apply_weights(\n"
    "        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None\n"
    "    ) -> torch.Tensor:\n"
    "        # " + GENESIS_PN118_MARKER + "\n"
    "        from vllm._genesis import int8_mlp_dispatch as _g118\n"
    "        _o = _g118.forward_int8(layer, x, bias)\n"
    "        if _o is not None:\n"
    "            return _o\n"
    "        return self.kernel.apply_weights(layer, x, bias)\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(
        "model_executor/layers/quantization/compressed_tensors/schemes/"
        "compressed_tensors_wNa16.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN118 MLP AWQ int4 -> INT8 per-canal",
        target_file=str(target),
        marker=GENESIS_PN118_MARKER,
        sub_patches=[TextPatch(name="pn118_wna16_int8",
                               anchor=ANCHOR_OLD,
                               replacement=ANCHOR_NEW,
                               required=True)],
        upstream_drift_markers=[],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN118")
    log_decision("PN118", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root no localizable"

    patcher = _make_patcher()
    if patcher is None:
        return "failed", "compressed_tensors_wNa16.py no encontrado"

    result, failure = patcher.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message=("MLP AWQ int4 -> INT8 per-canal en la carga, el int4 se "
                         "libera y el forward despacha por cutlass_scaled_mm"),
        patch_name=patcher.patch_name)


__all__ = ["apply", "GENESIS_PN118_MARKER", "ANCHOR_OLD", "ANCHOR_NEW"]
