# SPDX-License-Identifier: Apache-2.0
"""Arbol de borrador — hito 6: el cableado en el model runner v2 y en DFlash2.

Apagado por defecto (``GENESIS_ENABLE_ARBOL=1``). Se instala desde ``plugins_arranque`` (tiene
que correr en el proceso que sirve), por monkeypatch de clases: no hay anclas de texto.

La idea que ordena todo
-----------------------
El arbol existe SOLO entre la propuesta del borrador y la aceptacion. Apenas se acepta un
camino se COMPACTA todo lo que el paso dejo indexado por nodo (KV de atencion, estados ocultos,
estado conv, filas de cinta), y a partir de ahi el resto de vLLM — scheduler, post_update,
migracion de estado GDN, el borrador y su contexto — ve exactamente lo que veria con una
cadena de ``nacc`` tokens aceptados. Para el scheduler un arbol de 8 nodos son 8 tokens de
borrador como siempre: mismos slots, misma cuenta.

Los cinco enganches
-------------------
1. ``DFlash2Speculator._sample_path``  elige los 8 nodos (``arbol_borrador.construir`` +
   ``orden_dfs``) en vez de un camino. Corre adentro del grafo CUDA del borrador: forma fija.
2. ``DFlash2Speculator.propose``        guarda los padres por slot de req-state, al lado de
   donde vLLM guarda ``draft_tokens``.
3. ``model_state.prepare_inputs``       antes del forward del target: mascara de ancestros de
   PN131 y de GDN/conv, y posiciones de RoPE = base + PROFUNDIDAD (los slots de KV siguen
   siendo base + indice: ``input_batch.positions`` no se toca).
4. ``rejection_sample``                 aceptacion por recorrido (``arbol_borrador.aceptar``).
5. ``GPUModelRunner.sample``            despues de aceptar: compacta y repone las mascaras.
   La conv se compacta desde ``mamba_hybrid`` (gancho de PN122), justo antes de la migracion
   de estado de upstream, que lee esas columnas.

Cuando NO hay arbol (y por que sigue siendo correcto)
-----------------------------------------------------
La mascara de ancestros solo existe en el decode uniforme de PN131. Si el lote no es uniforme
(un prefill mezclado, un pedido con menos borradores, penalidades que asumen cadena) la
atencion corre causal. En preorden, los nodos 1..m hasta el primer salto de rama SON una cadena
(el camino goloso), asi que con mascara causal sus logits son exactos; los de despues no, y se
marcan inalcanzables para la aceptacion. El paso degrada a "cadena de m tokens": nunca a algo
incorrecto.

Todo en GPU. Las unicas decisiones en CPU usan arreglos que el runner ya tiene en numpy
(tokens agendados por pedido): no hay ninguna sincronizacion nueva.
"""

from __future__ import annotations

import logging
import os

import torch

from vllm._genesis import arbol_borrador as ab

log = logging.getLogger("genesis.arbol")

ACTIVO = os.environ.get("GENESIS_ENABLE_ARBOL", "0").strip().lower() in ("1", "true", "yes", "on")


class _Estado:
    listo = False
    K = 0                    # nodos = borradores por paso
    T = 0                    # K + 1: [ancla] + nodos
    padre_b = None           # [max_reqs, T]  por fila del lote: lo escribe el borrador (grafo)
    prof_b = None
    padre_v = None           # [slots + 1, T] por slot de req-state (la fila extra absorbe el -1)
    prof_v = None
    cadena_bits = None       # [T] int32
    ar_T = None              # [T] int64
    # del paso en curso
    uniforme = False
    camino = None            # [R, T-1] posiciones verificadas aceptadas
    nacc = None              # [R]
    aux = None               # estados ocultos auxiliares del paso (los consume el borrador)
    slots_por_capa = None
    capas_kv = None          # [(nombre, tensor KV)] de las capas de atencion del target
    kv_addrs = None
    kv_impl = None
    pasos = 0
    pasos_arbol = 0
    dup_padre = None
    dup_prof = None
    dup_bits = None
    bits_b = None            # [max_reqs, T] int32, del kernel constructor
    bits_v = None            # por slot: bits, delta de RoPE y "primera corrida" (todo precalculado
    delta_v = None           # al proponer: el paso siguiente solo copia)
    corrida_v = None
    todos = None
    filas = None


E = _Estado()

# Bisectar sin reiniciar (solo con GENESIS_ARBOL_DEBUG=1): bits leidos de /dev/shm/arbol_modo.
#   1 aceptar solo por la primera corrida   2 sin delta de RoPE        4 sin compactar KV
#   8 sin compactar estados ocultos         16 sin compactar la conv   32 volcar el paso al log
# Cambiarlo SOLO con el servidor ocioso: los dos ranks de TP tienen que leer lo mismo.
_DEBUG = os.environ.get("GENESIS_ARBOL_DEBUG", "0") == "1"
_modo_cache = [0, 0.0]


def _modo() -> int:
    if not _DEBUG:
        return 0
    import time
    ahora = time.monotonic()
    if ahora - _modo_cache[1] > 0.5:
        _modo_cache[1] = ahora
        try:
            with open("/dev/shm/arbol_modo") as f:
                _modo_cache[0] = int(f.read().strip() or 0)
        except (OSError, ValueError):
            _modo_cache[0] = 0
    return _modo_cache[0]


def _init(K: int, max_reqs: int, slots: int, dev) -> None:
    if E.listo:
        return
    E.K, E.T = K, K + 1
    cad = torch.arange(-1, K, dtype=torch.int32, device=dev)
    prof = torch.arange(0, K + 1, dtype=torch.int64, device=dev)
    E.padre_b = cad[None].repeat(max_reqs, 1).contiguous()
    E.prof_b = prof[None].repeat(max_reqs, 1).contiguous()
    E.padre_v = cad[None].repeat(slots + 1, 1).contiguous()
    E.prof_v = prof[None].repeat(slots + 1, 1).contiguous()
    E.cadena_bits = ((1 << torch.arange(K + 1, device=dev, dtype=torch.int32)) - 1)
    E.ar_T = prof.clone()
    E.bits_b = E.cadena_bits[None].repeat(max_reqs, 1).contiguous()
    E.bits_v = E.cadena_bits[None].repeat(slots + 1, 1).contiguous()
    E.delta_v = torch.zeros((slots + 1, K + 1), dtype=torch.int64, device=dev)
    E.corrida_v = torch.ones((slots + 1, K + 1), dtype=torch.bool, device=dev)
    E.todos = torch.ones((max_reqs, K + 1), dtype=torch.bool, device=dev)
    if K == 8:
        E.dup_padre = torch.tensor([-1, 0, 1, 2, 3, 0, 5, 6, 7], dtype=torch.int32, device=dev)
        E.dup_prof = torch.tensor([0, 1, 2, 3, 4, 1, 2, 3, 4], dtype=torch.int64, device=dev)
        E.dup_bits = ab.bits_ancestros(E.dup_padre[None])[0].contiguous()
    E.listo = True
    log.warning("[ARBOL] activo: %d nodos por paso, %d pedidos, %d slots", K, max_reqs, slots)


# ───────────────────────────── 1 y 2: el borrador ─────────────────────────────

_DUPLICADO = os.environ.get("GENESIS_ARBOL_PRUEBA_DUPLICADO", "0") == "1"
_path0 = None


def _sample_path(self, candidate_ids, scores, num_reqs):
    R = num_reqs
    if _DUPLICADO:
        # Diagnostico: la misma cadena de 4 tokens DOS veces, como dos ramas que salen del ancla.
        # Con un forward correcto los logits de los nodos 5..8 son los de 1..4 (mismo camino).
        _path0(self, candidate_ids, scores, num_reqs)
        self.draft_tokens[:R, 4:8] = self.draft_tokens[:R, 0:4]
        E.padre_b[:R] = E.dup_padre           # constantes creadas fuera del grafo
        E.prof_b[:R] = E.dup_prof
        E.bits_b[:R] = E.dup_bits
        return
    S, Kc = self.num_speculative_steps, self.selector_top_k
    ab.construir_dfs_kernel(candidate_ids.view(R, S, Kc).contiguous(),
                            scores.view(R, S, Kc, Kc).contiguous(),
                            self.draft_tokens[:R], E.padre_b[:R], E.prof_b[:R], S, E.bits_b[:R])
    # Los puntajes realizados (los lee la verificacion adaptativa y el cache de logits del
    # borrador, que la aceptacion en arbol no usa): se dejan definidos, con la fila del ancla.
    self._selector_scores[:R] = scores.view(R, S, Kc, Kc)[:, :, 0, :].to(self._selector_scores.dtype)


def _envolver_borrador() -> None:
    from vllm.v1.worker.gpu.spec_decode.dflash2 import speculator as m
    cls = m.DFlash2Speculator
    if getattr(cls, "_genesis_arbol", False):
        return
    global _path0
    init0, propose0, _path0 = cls.__init__, cls.propose, cls._sample_path

    def __init__(self, vllm_config, device):
        init0(self, vllm_config, device)
        slots = int(getattr(vllm_config.scheduler_config, "max_num_seqs", self.max_num_reqs))
        _init(int(self.num_speculative_steps), int(self.max_num_reqs),
              max(slots, int(self.max_num_reqs)) * 4, device)

    def propose(self, input_batch, *a, **kw):
        out = propose0(self, input_batch, *a, **kw)
        R = int(input_batch.num_reqs)
        idx = input_batch.idx_mapping[:R].long()          # un -1 cae en la fila extra
        pb = E.padre_b[:R]
        E.padre_v[idx] = pb
        E.prof_v[idx] = E.prof_b[:R]
        E.bits_v[idx] = E.bits_b[:R]
        E.delta_v[idx] = E.prof_b[:R] - E.ar_T[None]
        E.corrida_v[idx] = (pb.long() == (E.ar_T[None] - 1)).long().cumprod(dim=1).bool()
        return out

    cls.__init__, cls.propose, cls._sample_path = __init__, propose, _sample_path
    cls._genesis_arbol = True


# ───────────────────────────── 3: antes del forward ─────────────────────────────

def _hay_penalidades(runner, idx_np) -> bool:
    """Las penalidades de upstream cuentan como salida a los borradores ANTERIORES del lote, o sea
    que asumen cadena. Con ellas el paso va en cadena (exacto) en vez de aproximar."""
    try:
        ps = runner.sampler.penalties_state
        usa = getattr(ps, "use_penalty", None)
        if usa is None:
            return False
        usa = usa.np if hasattr(usa, "np") else usa
        return bool(usa[idx_np].any())
    except Exception:
        return False


def _antes_del_forward(runner, ms, input_batch) -> None:
    from vllm._genesis import gdn_cinta, sk18_attn
    E.uniforme, E.camino, E.nacc = False, None, None
    R, T = int(input_batch.num_reqs), E.T
    if R == 0 or not E.listo:
        gdn_cinta.paso_en_arbol(False)
        return
    nst = input_batch.num_scheduled_tokens[:R]
    dev = input_batch.idx_mapping.device
    anc131 = sk18_attn.mascara_arbol(dev, T)
    rope = getattr(ms, "rope_state", None)
    uniforme = bool((nst == T).all()) and anc131 is not None and R * T <= anc131.shape[0] \
        and rope is not None and not _hay_penalidades(runner, input_batch.idx_mapping_np[:R])
    gdn_cinta.paso_en_arbol(uniforme)
    E.pasos += 1
    if not uniforme:
        return
    E.pasos_arbol += 1
    idx = input_batch.idx_mapping[:R].long()
    bits = E.bits_v[idx].flatten()
    anc131[: R * T] = bits
    gdn_cinta.ancestros_gpu()[: R * T] = bits
    # RoPE: el nodo t va en la posicion base + profundidad(t), no base + t.
    delta = E.delta_v[idx].flatten()
    if _modo() & 2:
        delta = torch.zeros_like(delta)
    rope.positions[:, : R * T] += delta[None].to(rope.positions.dtype)
    E.uniforme = True


# ───────────────────────────── 4: la aceptacion ─────────────────────────────

_rejection0 = None

def _rejection_sample(target_logits, draft_logits, draft_sampled, cu_num_logits, pos, idx_mapping,
                      expanded_idx_mapping, expanded_local_pos, temperature, seed,
                      num_speculative_steps, synthetic_conditional_rates=None, use_fp64=False,
                      use_block_verification=False):
    if not E.listo:                       # el borrador no es DFlash2: lo de siempre
        return _rejection0(target_logits, draft_logits, draft_sampled, cu_num_logits, pos,
                           idx_mapping, expanded_idx_mapping, expanded_local_pos, temperature,
                           seed, num_speculative_steps, synthetic_conditional_rates,
                           use_fp64=use_fp64, use_block_verification=use_block_verification)
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
    R, T = cu_num_logits.shape[0] - 1, E.T
    idx = idx_mapping.long()
    en_arbol = E.uniforme and not (_modo() & 1)
    if en_arbol:
        # La semilla del ruido se indexa por posicion REAL (base + profundidad): asi el token j
        # del camino aceptado consume la misma clave que consumiria en una cadena. El lote es
        # uniforme, o sea que las filas de logits son R x T densas.
        pos = pos + E.delta_v[idx].flatten().to(pos.dtype)
        alcanzable = E.todos[:R]
    else:
        # sin mascara de arbol solo vale la primera corrida del preorden (ver el docstring)
        alcanzable = E.corrida_v[idx]
    muestra_f = gumbel_sample(target_logits, expanded_idx_mapping, temperature, seed, pos,
                              apply_temperature=False, is_drafting=False, use_fp64=use_fp64)
    pv = E.padre_v[idx]
    sampled, nacc, camino, E.filas = ab.aceptar_kernel(
        draft_sampled, muestra_f, cu_num_logits, pv, alcanzable, T, E.K)
    if _modo() & 32:
        token_v, muestra = draft_sampled[:T][None], muestra_f[:T][None]
    E.camino, E.nacc = camino, nacc
    if _modo() & 32:
        log.warning("[ARBOL dbg] unif=%s padre=%s tok=%s quiere=%s camino=%s nacc=%d",
                    E.uniforme, pv[0].tolist(), token_v[0].tolist(), muestra[0].tolist(),
                    camino[0].tolist(), int(nacc[0]))
    return sampled[:, : num_speculative_steps + 1].contiguous(), nacc


# ───────────────────────────── 5: despues de aceptar ─────────────────────────────

def _capas_kv(runner):
    """Capas de atencion completa del TARGET (las del borrador tienen su propio contexto, que el
    borrador recalcula desde los estados ocultos ya compactados)."""
    if E.capas_kv is not None:
        return E.capas_kv
    ctx = runner.vllm_config.compilation_config.static_forward_context
    nombres = []
    for g in runner.kv_cache_config.kv_cache_groups:
        spec = g.kv_cache_spec
        if type(spec).__name__ == "FullAttentionSpec" and not getattr(spec, "sliding_window", None):
            nombres += list(g.layer_names)
    from vllm._genesis import sk18_attn
    capas = []
    for nm in nombres:
        capa = ctx.get(nm)
        impl = getattr(capa, "impl", None)
        if capa is None or impl is None or not sk18_attn.activo(impl, capa):
            continue
        kv = capa.kv_cache
        kv = kv[0] if isinstance(kv, (list, tuple)) else kv
        if kv.dim() == 4:
            capas.append((nm, kv))
    E.capas_kv = capas
    log.warning("[ARBOL] compactacion de KV sobre %d capas de atencion del target", len(capas))
    return capas


def _compactar(runner, hidden_states, input_batch) -> None:
    from vllm._genesis import gdn_cinta
    R, T = int(input_batch.num_reqs), E.T
    dev = hidden_states.device
    cam, nacc = E.camino[:R], E.nacc[:R]
    # filas de cinta del camino, para el GDN del paso que viene y para la conv (slot = req + 1)
    cgpu = gdn_cinta.camino_gpu()
    if cgpu is not None:
        cgpu[input_batch.idx_mapping[:R].long() + 1] = E.filas[:R]
    if not E.uniforme:
        return                        # se acepto por la primera corrida: ya es una cadena
    bos = (torch.arange(R, device=dev) * T)[:, None]
    j = torch.arange(T - 1, device=dev)[None]
    vale = j < (nacc - 1)[:, None]
    dst = (bos + 1 + j)
    src = torch.where(vale, bos + cam.clamp(min=0), dst)             # lo invalido se copia a si mismo
    dst, src = dst.flatten(), src.flatten()
    # estados ocultos que consume el borrador (src >= dst y crecientes: index_copy lee antes)
    if not (_modo() & 8):
        hidden_states[dst] = hidden_states[src]
        for h in (E.aux or ()):
            h[dst] = h[src]
    if _modo() & 4:
        return
    # KV de atencion: el nodo p_j se escribio en el slot base + p_j y tiene que quedar en base + j.
    # La clave ya lleva el RoPE de base + profundidad = base + j: la copia es exacta.
    capas = _capas_kv(runner)
    if not capas:
        return
    # Todas las capas de atencion del target comparten grupo de KV: un solo slot mapping.
    sm = E.slots_por_capa.get(capas[0][0]) if E.slots_por_capa else None
    if sm is None:
        return
    from vllm._genesis import sk18_attn
    if E.kv_addrs is None:
        E.kv_addrs = torch.tensor([kv.data_ptr() for _, kv in capas], dtype=torch.int64, device=dev)
        E.kv_impl = runner.vllm_config.compilation_config.static_forward_context[capas[0][0]].impl
    sk18_attn.compactar_slots([kv for _, kv in capas], E.kv_addrs,
                              sm[src].long().view(R, T - 1).contiguous(),
                              sm[dst].long().view(R, T - 1).contiguous(), E.kv_impl)


def _envolver_runner() -> None:
    from vllm.v1.worker.gpu import model_runner as mr
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler as rs
    cls = mr.GPUModelRunner
    if getattr(cls, "_genesis_arbol", False):
        return
    global _rejection0
    sample0, tokens0, exec0 = cls.sample, cls.sample_tokens, cls.execute_model
    _rejection0 = rs.rejection_sample
    rs.rejection_sample = _rejection_sample

    def execute_model(self, *a, **kw):
        ms = getattr(self, "model_state", None)
        if ms is not None:
            ms._genesis_runner = self
        return exec0(self, *a, **kw)

    def sample_tokens(self, grammar_output):
        st = self.execute_model_state
        if st is not None:
            E.aux = st.aux_hidden_states
            E.slots_por_capa = st.slot_mappings_by_layer
        return tokens0(self, grammar_output)

    def sample(self, hidden_states, input_batch, grammar_output):
        from vllm._genesis import gdn_cinta, sk18_attn
        out = sample0(self, hidden_states, input_batch, grammar_output)
        if E.listo:
            if E.camino is not None:
                _compactar(self, hidden_states, input_batch)
            # el borrador comparte los buffers de PN131 y no verifica ningun arbol
            R = int(input_batch.num_reqs)
            if E.uniforme and R:
                m = sk18_attn.mascara_arbol(hidden_states.device, E.T)
                m[: R * E.T] = E.cadena_bits.repeat(R)
                gdn_cinta.ancestros_gpu()[: R * E.T] = E.cadena_bits.repeat(R)
            if E.pasos and E.pasos % 2000 == 0:
                log.warning("[ARBOL] %d pasos, %d en arbol", E.pasos, E.pasos_arbol)
        return out

    cls.sample, cls.sample_tokens, cls.execute_model = sample, sample_tokens, execute_model
    cls._genesis_arbol = True

    # model_state se crea mas tarde (load_model): se envuelve la clase la primera vez que se ve
    from vllm.v1.worker.gpu.model_states import mamba_hybrid as mh
    msc = mh.MambaHybridModelState
    if not getattr(msc, "_genesis_arbol", False):
        prep0 = msc.prepare_inputs

        def prepare_inputs(self, input_batch, req_states):
            out = prep0(self, input_batch, req_states)
            runner = getattr(self, "_genesis_runner", None)
            _antes_del_forward(runner, self, input_batch)
            return out

        post0 = msc.postprocess_state

        def postprocess_state(self, idx_mapping, num_sampled, num_computed_tokens=None):
            # La conv se compacta ANTES de la migracion de upstream, que lee esas columnas.
            if E.uniforme and E.camino is not None and not isinstance(num_sampled, int) \
                    and not (_modo() & 16):
                from vllm._genesis import arbol_conv, gdn_cinta
                capas = gdn_cinta.capas_gdn()
                if capas:
                    arbol_conv.compactar_v2(
                        self._mamba_ctx, capas, gdn_cinta.grupos_gdn(), self._mamba_state_idx_gpu,
                        idx_mapping, num_sampled, gdn_cinta.camino_gpu(), idx_mapping.shape[0])
            return post0(self, idx_mapping, num_sampled, num_computed_tokens)

        msc.prepare_inputs, msc.postprocess_state = prepare_inputs, postprocess_state
        msc._genesis_arbol = True


# ───────────────────────────── conv: forward y compactacion ─────────────────────────────

def _envolver_conv() -> None:
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as m
    if getattr(m, "_genesis_arbol", False):
        return
    from vllm._genesis import arbol_conv, gdn_cinta
    conv0 = m.causal_conv1d_update

    def causal_conv1d_update(x, conv_state, weight, bias=None, activation=None, **kw):
        nacc, qsl = kw.get("num_accepted_tokens"), kw.get("query_start_loc")
        anc = gdn_cinta.ancestros_gpu()
        if nacc is None or qsl is None or anc is None or bias is not None \
                or not gdn_cinta.paso_arbol_activo():
            return conv0(x, conv_state, weight, bias, activation, **kw)
        o = arbol_conv.salidas(x, conv_state, weight, activation, kw["conv_state_indices"],
                               nacc, qsl, anc)
        res = conv0(x, conv_state, weight, bias, activation, **kw)      # escribe el estado
        res.copy_(o)
        return res

    m.causal_conv1d_update = causal_conv1d_update
    m._genesis_arbol = True


def instalar() -> None:
    if not ACTIVO:
        return
    if os.environ.get("GENESIS_ENABLE_PN122_GDN_CINTA", "0") != "1":
        log.error("[ARBOL] necesita PN122 (la cinta de GDN): queda APAGADO")
        return
    _envolver_borrador()
    _envolver_runner()
    _envolver_conv()
    log.warning("[ARBOL] enganches instalados (borrador, runner v2, conv de GDN)")


__all__ = ["instalar", "ACTIVO", "E"]
