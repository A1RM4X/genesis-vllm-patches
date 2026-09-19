# SPDX-License-Identifier: Apache-2.0
"""Diagnostico del borrador DFlash2: que propone, con que entrada, y con que pesos.

Con `num_accepted_tokens_total = 0` y ningun error, los contadores no alcanzan para
distinguir dos causas muy distintas:

  * los PESOS del borrador estan mal (cuantizacion, carga, repack) -> propone basura;
  * el CABLEADO esta mal (hidden states auxiliares de las capas equivocadas, embedding
    compartido que no llega) -> propone texto plausible pero corrido.

Las dos dan la misma metrica. Lo unico que las separa es mirar los ids. La primera corrida
mostro algo que no es ninguna de las dos: el borrador propone SIEMPRE el mismo id, siete
veces, en todos los pasos. Eso es logits constantes — la salida no depende de la entrada — y
manda a mirar mas atras: la norma del tensor que entra, la del que sale de `fc`, y la de los
pesos que deberia compartir con el target (embed_tokens / lm_head, que el checkpoint del
borrador NO trae).

Se prende con GENESIS_DIAG_DFLASH=1 y se apaga solo a los pocos pasos: corre adentro del
lazo de decode y no puede quedar imprimiendo por request.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.diag_dflash")

_PASOS = int(os.environ.get("GENESIS_DIAG_DFLASH_PASOS", "4"))


def _stats(t) -> str:
    """Forma, dtype y magnitud de un tensor. Un `norma=0` es el hallazgo que se busca."""
    if t is None:
        return "None"
    try:
        import torch

        if isinstance(t, (list, tuple)):
            return "%d x %s" % (len(t), _stats(t[0]))
        if not isinstance(t, torch.Tensor):
            return repr(t)[:60]
        f = t.detach().float()
        return "%s %s norma=%.4g min=%.4g max=%.4g nan=%d" % (
            tuple(t.shape), str(t.dtype).replace("torch.", ""),
            float(f.norm()), float(f.min()), float(f.max()),
            int(f.isnan().sum()))
    except Exception as e:                                       # noqa: BLE001
        return f"<{type(e).__name__}>"


def _pesos_compartidos(self) -> str:
    """embed_tokens y lm_head: el checkpoint del borrador no los trae, los toma del target.

    Si esa union no se hizo, el borrador arranca de un embedding sin inicializar (o en cero)
    y su salida deja de depender del texto — exactamente el sintoma de logits constantes.
    """
    partes = []
    try:
        modelo = getattr(self, "model", None)
        for ruta in ("model.embed_tokens.weight", "model.model.embed_tokens.weight",
                     "lm_head.weight", "model.lm_head.weight", "model.fc.weight",
                     "model.fc.weight_packed", "model.norm.weight"):
            obj = modelo
            for pieza in ruta.split("."):
                obj = getattr(obj, pieza, None)
                if obj is None:
                    break
            if obj is not None:
                partes.append("%s: %s" % (ruta, _stats(obj)))
    except Exception as e:                                       # noqa: BLE001
        partes.append(f"<{type(e).__name__}: {e}>")
    return " | ".join(partes) if partes else "no encontre ninguno"


def cazar_nan(modelo, etiqueta: str = "borrador") -> None:
    """Engancha TODOS los submodulos y denuncia el PRIMERO que saca NaN con entrada finita.

    Por que hace falta en vez de razonar: el borrador termina escupiendo NaN en fp16 y anda
    en bf16, y es facil armar una historia convincente sobre donde desborda. La primera que
    arme (la suma de cuadrados de `hidden_norm`, con picos de 12.000 al cuadrado pasandose de
    65.504) era falsa: `ops.rms_norm` acumula la varianza en fp32. Sin saber el modulo exacto,
    "arreglarlo" es adivinar.

    OJO: esto SINCRONIZA en cada modulo. Solo sirve con los grafos CUDA apagados
    (--enforce-eager) y para una corrida de diagnostico.
    """
    import torch

    # Traza de la PRIMERA pasada, en orden de ejecucion y con los modulos limpios tambien.
    # Mirar solo los que ensucian no alcanza: la primera corrida mostro layer 1 recibiendo un
    # (1) NaN sin que ningun modulo de layer 0 lo hubiera producido, porque lo que ensucia
    # puede ser una op que NO es un modulo (una suma residual, una conv funcional) o un
    # buffer precargado. Con la cuenta de NaN entrando y saliendo de cada modulo en orden, el
    # salto de 0 a 1 queda entre dos lineas concretas.
    corridas = {"n": 0}
    TOPE = int(os.environ.get("GENESIS_DIAG_DFLASH_TRAZA", "260"))

    def _cuenta(x) -> int:
        if isinstance(x, torch.Tensor) and x.is_floating_point():
            return int(torch.isnan(x).sum() + torch.isinf(x).sum())
        if isinstance(x, (list, tuple)):
            return sum(_cuenta(y) for y in x)
        if isinstance(x, dict):
            return sum(_cuenta(y) for y in x.values())
        return 0

    def _hook(nombre):
        def f(_mod, entrada, kw, salida):
            if corridas["n"] >= TOPE:
                return
            corridas["n"] += 1
            ent = _cuenta(entrada) + _cuenta(kw)
            sal = _cuenta(salida)
            log.warning("[traza %s] %3d %-52s NaN entra=%d sale=%d | %s",
                        etiqueta, corridas["n"], nombre, ent, sal,
                        _stats(salida[0] if isinstance(salida, tuple) and salida else salida))
        return f

    n = 0
    for nombre, mod in modelo.named_modules():
        if nombre:
            mod.register_forward_hook(_hook(nombre), with_kwargs=True)
            n += 1
    log.warning("[NaN %s] cazador puesto sobre %d submodulos", etiqueta, n)


def enganchar() -> None:
    """Envuelve ``propose`` de los especuladores DFlash. No puede tumbar el arranque."""
    import inspect

    import torch

    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (  # noqa: PLC0415
        DFlashSpeculator,
    )

    original = DFlashSpeculator.propose
    firma = inspect.signature(original)
    estado = {"n": 0, "pesos": False}

    def propose(self, *a, **kw):
        salida = original(self, *a, **kw)
        if kw.get("dummy_run") or estado["n"] >= _PASOS:
            return salida
        try:
            estado["n"] += 1
            atado = firma.bind(self, *a, **kw)
            atado.apply_defaults()
            arg = atado.arguments

            if not estado["pesos"]:
                estado["pesos"] = True
                log.warning("[DIAG dflash] pesos compartidos -> %s", _pesos_compartidos(self))
                if os.environ.get("GENESIS_DIAG_DFLASH_NAN") == "1":
                    # Recien aca: antes de la primera pasada real el modelo todavia no
                    # existe. Los hooks entran en vigor desde la llamada siguiente.
                    cazar_nan(self.model, "borrador")

            fila = (salida[0].tolist()
                    if isinstance(salida, torch.Tensor) and salida.numel() else [])
            ult = arg.get("last_sampled")
            prev = int(ult.flatten()[0]) if ult is not None and ult.numel() else -1
            log.warning(
                "[DIAG dflash %d/%d] aux=%s | hidden=%s | target id=%d | propone %s",
                estado["n"], _PASOS,
                _stats(arg.get("aux_hidden_states")),
                _stats(arg.get("last_hidden_states")),
                prev, fila)
        except Exception as e:                                   # noqa: BLE001
            log.warning("[DIAG dflash] no se pudo volcar (%s: %s)", type(e).__name__, e)
        return salida

    DFlashSpeculator.propose = propose
    _enganchar_fc()
    log.warning("[DIAG dflash] enganchado sobre DFlashSpeculator.propose (%d pasos)", _PASOS)


def _enganchar_fc() -> None:
    """``combine_hidden_states`` es la PRIMERA etapa cuantizada del borrador.

    Toma los 5 hidden states del target concatenados (25600 features) y los proyecta a 5120
    con ``fc``, que en este checkpoint es W4A16 empaquetado y pasa por Marlin. Si la salida
    de aca ya es constante, cero o NaN mientras la entrada esta sana, el problema esta en el
    GEMM cuantizado del borrador y no en el cableado que lo alimenta.
    """
    try:
        from vllm.model_executor.models import qwen3_dflash
    except Exception as e:                                       # noqa: BLE001
        log.warning("[DIAG dflash] no pude importar qwen3_dflash (%s)", type(e).__name__)
        return

    clases = [v for v in vars(qwen3_dflash).values()
              if isinstance(v, type) and "combine_hidden_states" in vars(v)]
    if not clases:
        log.warning("[DIAG dflash] ninguna clase de qwen3_dflash define combine_hidden_states")
        return

    for cls in clases:
        original = cls.combine_hidden_states
        estado = {"n": 0}

        def combine(self, hidden_states, _orig=original, _e=estado, _c=cls.__name__):
            salida = _orig(self, hidden_states)
            # El calentamiento entra con un tensor de ceros: no dice nada, y se come el cupo.
            vacio = bool(hidden_states is not None and float(hidden_states.abs().max()) == 0.0)
            if _e["n"] < _PASOS and not vacio:
                _e["n"] += 1
                log.warning("[DIAG dflash fc %d/%d en %s] entra %s -> sale %s",
                            _e["n"], _PASOS, _c, _stats(hidden_states), _stats(salida))
            return salida

        cls.combine_hidden_states = combine
        log.warning("[DIAG dflash] enganchado %s.combine_hidden_states", cls.__name__)

    _enganchar_candidatos()


def _enganchar_candidatos() -> None:
    """``compute_candidates`` es el top-k del lm_head del TARGET, no del borrador.

    DFlash2 no propone por argmax: propone entre los candidatos que saca el lm_head del
    target via ``get_top_k_tokens``, y despues el selector con sus codebooks ordena. El
    README del checkpoint avisa que con un lm_head CUANTIZADO ese camino necesita parche
    ("upstream refuses a non-bf16 lm_head for the candidate top-k") — y el nuestro es
    W4A16 con Marlin. Si los ids que salen de aca ya son constantes, el borrador no tiene
    nada que hacer y la culpa no es suya.
    """
    try:
        from vllm.model_executor.models import qwen3_dflash2
    except Exception as e:                                       # noqa: BLE001
        log.warning("[DIAG dflash] no pude importar qwen3_dflash2 (%s)", type(e).__name__)
        return

    clases = [v for v in vars(qwen3_dflash2).values()
              if isinstance(v, type) and "compute_candidates" in vars(v)]
    for cls in clases:
        original = cls.compute_candidates
        estado = {"n": 0}

        def candidatos(self, hidden_states, _orig=original, _e=estado, _c=cls.__name__):
            ids, valores = _orig(self, hidden_states)
            vacio = bool(hidden_states is not None and float(hidden_states.abs().max()) == 0.0)
            if _e["n"] < _PASOS and not vacio:
                _e["n"] += 1
                try:
                    muestra = ids[0].tolist()[:8]
                except Exception:                                # noqa: BLE001
                    muestra = "?"
                log.warning(
                    "[DIAG dflash candidatos %d/%d en %s] hidden %s -> ids %s | valores %s "
                    "| primera fila %s",
                    _e["n"], _PASOS, _c, _stats(hidden_states), _stats(ids), _stats(valores),
                    muestra)
            return ids, valores

        cls.compute_candidates = candidatos
        log.warning("[DIAG dflash] enganchado %s.compute_candidates", cls.__name__)
