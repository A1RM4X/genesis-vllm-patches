# SPDX-License-Identifier: Apache-2.0
"""PN132 — el borrador MTP propone sobre un vocabulario RECORTADO (estilo FR-Spec).

Por que
-------
El vocabulario del modelo tiene 248.320 entradas. En cada paso de decode el borrador MTP
proyecta contra TODAS, una vez por token propuesto (4 con K=4): medido, 3,74 ms de los 29,6 ms
del paso (12,6 %). La verificacion la hace el modelo grande con el vocabulario completo, asi que
el borrador no necesita poder proponer cualquier token: le alcanza con los frecuentes.

Que hace
--------
Arma una lista de los ``GENESIS_PN132_VOCAB`` tokens mas frecuentes en un corpus de calibracion
(mas los especiales), guarda una copia propia de esas filas del ``lm_head`` y calcula los logits
del borrador con esa matriz. Devuelve un tensor del ancho completo con -inf afuera del
subconjunto, asi que el muestreo y el rechazo de MTP siguen funcionando sin cambios:
la distribucion propuesta q' es otra propuesta valida (el muestreo por rechazo corrige contra
la distribucion del modelo grande, que no se toca).

Costo: una matriz de V' x hidden en fp16 por GPU (32k -> 327 MB; 16k -> 164 MB).
Se apaga con ``GENESIS_ENABLE_PN132_VOCAB=0``.
"""

from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("genesis.pn132")

VOCAB_SUB = int(os.environ.get("GENESIS_PN132_VOCAB", 32768))
ESPECIALES = int(os.environ.get("GENESIS_PN132_ESPECIALES", 2048))   # primeros ids, siempre dentro
_estado: dict[int, dict] = {}
_aviso = set()


def activo() -> bool:
    return os.environ.get("GENESIS_ENABLE_PN132_VOCAB", "0") == "1"


def _corpus() -> str:
    """Texto de calibracion: lo que haya de los corpus del proyecto, si no un fallback chico."""
    import glob
    trozos = []
    for p in ("/p/calib_code.txt", "/p/calib_docs.txt", "/etc/qwen-froggeric-chat-template.jinja"):
        try:
            with open(p, errors="ignore") as f:
                trozos.append(f.read(8_000_000))
        except OSError:
            pass
    # Ademas, codigo real (el del propio servidor): son varios MB y cubren el vocabulario que
    # aparece de verdad en la carga de trabajo. Con poco corpus el recorte deja de tener sentido:
    # los ids sin apariciones entran por orden y el borrador no puede proponer lo que hace falta.
    for f in sorted(glob.glob("/usr/local/lib/python3.12/dist-packages/vllm/**/*.py", recursive=True)):
        try:
            with open(f, errors="ignore") as h:
                trozos.append(h.read())
        except OSError:
            pass
    return "\n".join(trozos)


def _ids(dev, vocab_size: int) -> torch.Tensor:
    """Ids del subconjunto: especiales + los mas frecuentes del corpus. Cacheado en disco."""
    cache = f"/root/.cache/vllm/genesis/pn132_ids_{VOCAB_SUB}_{vocab_size}.pt"
    try:
        ids = torch.load(cache, map_location="cpu")
        if ids.numel() >= VOCAB_SUB // 2:
            return ids.to(dev)
    except Exception:
        pass
    cuenta = torch.zeros(vocab_size, dtype=torch.int64)
    try:
        import glob
        from tokenizers import Tokenizer
        # El worker no tiene el vllm_config seteado en este punto: el tokenizer se busca en el
        # cache de HuggingFace montado (el tokenizer.json mas grande es el del modelo servido).
        tok = None
        for c in sorted(glob.glob("/root/.cache/huggingface/hub/models--*/snapshots/*/tokenizer.json"),
                        key=lambda f: os.path.getsize(f), reverse=True):
            t = Tokenizer.from_file(c)
            if abs(t.get_vocab_size() - vocab_size) <= 4096:    # el del modelo servido
                tok, ruta_tok = t, c
                break
        if tok is None:
            raise FileNotFoundError("ningun tokenizer del cache coincide con el vocabulario")
        texto = _corpus()
        for i in range(0, len(texto), 200_000):
            ids_trozo = tok.encode(texto[i:i + 200_000], add_special_tokens=False).ids
            t = torch.tensor([x for x in ids_trozo if x < vocab_size], dtype=torch.int64)
            if t.numel():
                cuenta.scatter_add_(0, t, torch.ones_like(t))
        log.warning("PN132: corpus tokenizado con %s (%d caracteres)", ruta_tok.split("/")[-3], len(texto))
    except Exception as e:                       # sin tokenizer: solo los primeros ids
        log.warning("PN132: no se pudo tokenizar el corpus (%s); uso los ids bajos", e)
    cuenta[:ESPECIALES] = torch.iinfo(torch.int64).max      # especiales siempre adentro
    ids = torch.argsort(cuenta, descending=True)[:VOCAB_SUB].sort().values.contiguous()
    vistos = int((cuenta[ids] > 0).sum())
    log.warning("PN132: vocabulario del borrador %d de %d (%d con apariciones en el corpus)",
                ids.numel(), vocab_size, vistos)
    try:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        torch.save(ids, cache)
    except Exception:
        pass
    return ids.to(dev)


def _del_checkpoint(ids, dev, vocab, hidden_esp):
    """Filas del lm_head leidas del checkpoint (ahi estan en fp16/bf16; en memoria vLLM las
    tiene empaquetadas en int8 con un layout propio que no conviene destripar)."""
    import glob
    import json
    from safetensors import safe_open
    elegido = None
    for idxf in sorted(glob.glob("/root/.cache/huggingface/hub/models--*/snapshots/*/model.safetensors.index.json"),
                       key=lambda f: os.path.getsize(f), reverse=True):
        mapa = json.load(open(idxf))["weight_map"]
        clave = next((k for k in mapa if k.endswith("lm_head.weight")), None)
        if clave is None:
            continue
        ruta = os.path.join(os.path.dirname(idxf), mapa[clave])
        try:
            with safe_open(ruta, framework="pt", device="cpu") as f:
                forma = f.get_slice(clave).get_shape()
        except Exception:
            continue
        log.warning("PN132: candidato %s %s", ruta.split("/")[-3][:44], forma)
        if len(forma) == 2 and forma[0] == vocab and forma[1] == hidden_esp:   # el modelo servido
            elegido = (ruta, clave)
            break
    if elegido is None:
        raise FileNotFoundError(
            f"ningun checkpoint del cache tiene lm_head.weight de {vocab} x {hidden_esp}")
    ruta, clave = elegido
    ids_cpu = ids.cpu()
    trozos = []
    with safe_open(ruta, framework="pt", device="cpu") as f:
        sl = f.get_slice(clave)
        n_total, hidden = sl.get_shape()
        paso = 8192
        for a in range(0, n_total, paso):
            b = min(a + paso, n_total)
            sel = ids_cpu[(ids_cpu >= a) & (ids_cpu < b)] - a
            if sel.numel() == 0:
                continue
            bloque = sl[a:b, :]
            trozos.append(bloque.index_select(0, sel).to(torch.float16))
    return torch.cat(trozos, 0).to(dev), hidden


def _pesos(cabeza, ids, dev, vocab, hidden_esp):
    """Devuelve (W_sub, None). Primero prueba el tensor denso; si esta cuantizado, va al checkpoint."""
    w = getattr(cabeza, "weight", None)
    if w is not None and w.dim() == 2 and w.dtype in (torch.float16, torch.bfloat16, torch.float32):
        n_local = w.shape[0]
        if n_local < int(ids.max()) + 1:
            ini = getattr(cabeza, "vocab_start_index", 0) or 0
            loc = ids - ini
            dentro = (loc >= 0) & (loc < n_local)
            return w.index_select(0, loc[dentro].to(w.device)).to(dev, torch.float16), dentro
        return w.index_select(0, ids.to(w.device)).to(dev, torch.float16), None
    try:
        t0 = __import__("time").time()
        sub, hidden = _del_checkpoint(ids, dev, vocab, hidden_esp)
        log.warning("PN132: cabeza del borrador leida del checkpoint: %s en %.1f s",
                    tuple(sub.shape), __import__("time").time() - t0)
        return sub, None
    except Exception as e:  # noqa: BLE001
        log.warning("PN132: no se pudo armar la cabeza recortada (%s); PN132 no se aplica", e)
        return None


def logits(modelo, hidden_states: torch.Tensor):
    """Logits del borrador sobre el subconjunto, en un tensor del ancho completo."""
    dev = hidden_states.device
    st = _estado.get(dev.index)
    if st is None and torch.cuda.is_current_stream_capturing():
        return None          # nunca reservar memoria adentro de la captura de un CUDA graph
    if st is None:
        cabeza = getattr(modelo, "lm_head", None)
        if cabeza is None:
            return None
        vocab = int(getattr(cabeza, "org_vocab_size", 0) or getattr(cabeza, "num_embeddings", 0)
                    or getattr(cabeza, "weight", torch.empty(0)).shape[0])
        log.warning("PN132: lm_head=%s vocab=%d attrs=(org=%s num=%s start=%s)", type(cabeza).__name__,
                    vocab, getattr(cabeza, "org_vocab_size", None), getattr(cabeza, "num_embeddings", None),
                    getattr(cabeza, "vocab_start_index", None))
        ids = _ids(dev, vocab)
        r = _pesos(cabeza, ids, dev, vocab, int(hidden_states.shape[-1]))
        if r is None:
            _estado[dev.index] = {"off": True}
            st = _estado[dev.index]
        else:
            w, dentro = r
            if dentro is not None:
                ids = ids[dentro]
            st = _estado[dev.index] = {"off": False, "w": w, "ids": ids.to(dev), "vocab": vocab}
            log.warning("PN132: cabeza del borrador %s (%.0f MiB), vocabulario completo %d",
                        tuple(w.shape), w.numel() * 2 / 2**20, vocab)
    if st.get("off"):
        return modelo.logits_processor(modelo.lm_head, hidden_states)
    h = hidden_states if hidden_states.dim() == 2 else hidden_states.view(-1, hidden_states.shape[-1])
    sub = torch.nn.functional.linear(h.to(st["w"].dtype), st["w"])          # [n, V']
    out = torch.full((sub.shape[0], st["vocab"]), float("-inf"),
                     dtype=torch.float32, device=sub.device)
    out.index_copy_(1, st["ids"], sub.float())
    return out
