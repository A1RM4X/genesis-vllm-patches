# SPDX-License-Identifier: Apache-2.0
"""DIAGNOSTICO — volcado de las distribuciones por posicion del borrador DFlash.

Para estimar OFFLINE cuanto rendiria verificar un ARBOL (DDTree, arXiv 2604.12989; con
presupuesto por costo, CaDDTree, arXiv 2606.01813) en vez de una cadena, ANTES de escribir un
solo kernel. El arbol se arma con las distribuciones por posicion que el borrador ya calcula en
su unica pasada; aca solo se las guarda.

``GENESIS_DIAG_DDTREE_DUMP=<dir>`` envuelve ``Speculator.sample_draft``: por cada paso escribe una
linea JSON con, para cada (pedido, posicion predicha P), el top-``GENESIS_DIAG_DDTREE_TOPK`` de
ids y probabilidades. Solo el rank 0 escribe. Recalcula los logits (una pasada extra del lm_head
del borrador) y sincroniza para bajarlos a CPU: es una regla de medir, NO va en produccion.

La "verdad" no se vuelca aca: con greedy es la respuesta final, que la API devuelve con
``return_token_ids``. El simulador es ``tests/bench/medicion/ddtree_sim.py``.
"""

from __future__ import annotations

import json
import logging
import os

import torch

log = logging.getLogger("genesis.diag.ddtree")

_archivo = None
_n = 0


def _rank() -> int:
    try:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())
    except Exception:                                        # noqa: BLE001
        return 0


def enganchar() -> None:
    """DFlash2 ya tiene lo que un arbol necesita: por posicion, ``selector_top_k`` candidatos y una
    tabla de puntajes de TRANSICION ``scores[paso, candidato previo, candidato]`` (el selector es
    de primer orden: condiciona en el candidato elegido en la posicion anterior). Hoy se recorre
    de forma golosa: un solo camino. Aca solo se guarda la tabla entera.

    Tres enganches, porque TODO ``_generate_draft`` va dentro del CUDA graph y en el replay no
    corre nada de Python:

    * ``capture``      reserva los buffers estaticos ANTES de capturar;
    * ``_sample_path`` (dentro del grafo) copia candidatos y puntajes — solo ops capturables;
    * ``propose``      (fuera del grafo, una vez por paso) los baja a CPU y escribe la linea.

    No se mide en eager a proposito: ese camino acepta distinto.
    """
    from vllm.v1.worker.gpu.spec_decode.dflash2 import speculator as d2

    cls = d2.DFlash2Speculator
    if getattr(cls._sample_path, "_genesis_ddtree", False):
        return
    capture_o, path_o, propose_o = cls.capture, cls._sample_path, cls.propose
    destino = os.environ["GENESIS_DIAG_DDTREE_DUMP"]

    def _reservar(self):
        if getattr(self, "_g_ddt_sc", None) is None:
            R, S, K = int(self.max_num_reqs), int(self.num_speculative_steps), int(self.selector_top_k)
            dev = self._selector_scores.device
            self._g_ddt_sc = torch.zeros(R, S, K, K, dtype=torch.float32, device=dev)
            self._g_ddt_id = torch.zeros(R, S, K, dtype=torch.int64, device=dev)

    def capture(self, *a, **kw):
        _reservar(self)
        return capture_o(self, *a, **kw)

    def _sample_path(self, candidate_ids, scores, num_reqs):
        if getattr(self, "_g_ddt_sc", None) is None and not torch.cuda.is_current_stream_capturing():
            _reservar(self)
        if getattr(self, "_g_ddt_sc", None) is not None:
            S, K = int(self.num_speculative_steps), int(self.selector_top_k)
            self._g_ddt_sc[:num_reqs].copy_(scores.reshape(num_reqs, S, K, K).float())
            self._g_ddt_id[:num_reqs].copy_(candidate_ids.reshape(num_reqs, S, K))
        return path_o(self, candidate_ids, scores, num_reqs)

    def propose(self, *a, **kw):
        global _archivo, _n
        out = propose_o(self, *a, **kw)
        try:
            if _rank() == 0 and getattr(self, "_g_ddt_sc", None) is not None and out is not None:
                nr = int(out.shape[0])
                S = int(self.num_speculative_steps)
                if nr > 0:
                    if _archivo is None:
                        os.makedirs(destino, exist_ok=True)
                        _archivo = open(os.path.join(destino, f"dump_{os.getpid()}.jsonl"), "a",
                                        buffering=1 << 20)
                        log.warning("[DIAG ddtree] volcando a %s", _archivo.name)
                    _archivo.write(json.dumps({
                        "pos": self.sample_pos[: nr * S].view(nr, S).tolist(),   # posicion P predicha
                        "req": self.sample_idx_mapping[: nr * S].view(nr, S)[:, 0].tolist(),
                        "borrador": out.reshape(nr, -1).tolist(),
                        "cand": self._g_ddt_id[:nr].tolist(),
                        "sc": [[[[round(x, 3) for x in f] for f in paso] for paso in r]
                               for r in self._g_ddt_sc[:nr].tolist()],
                    }) + "\n")
                    _n += 1
                    if _n % 20 == 0:
                        _archivo.flush()
        except Exception as e:                                # noqa: BLE001
            log.error("[DIAG ddtree] fallo el volcado (%s: %s)", type(e).__name__, e)
        return out

    _sample_path._genesis_ddtree = True
    cls.capture, cls._sample_path, cls.propose = capture, _sample_path, propose
    log.warning("[DIAG ddtree] enganchado en DFlash2Speculator (capture + _sample_path + propose)")
