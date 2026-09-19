# SPDX-License-Identifier: Apache-2.0
"""Genesis PN122: rollback del MTP en GDN con CINTA en vez de K copias del estado.

Por qué
-------
Medido con opencode (status de PN115, ``bloques_por_request``): con MTP K=3 cada
request ocupa 5-7 bloques en CADA uno de los 3 grupos GDN, ~18 bloques = ~15k
tokens de KV. La causa es ``MambaSpec.num_speculative_blocks = K``: el kernel de
decode spec escribe el estado completo (786 KB por capa por rank) tras cada uno
de los K+1 tokens, y el pool reserva un bloque por cada una de esas copias.

Qué hace
--------
La actualización de GDN es de rango 1::

    S <- S * exp(g) ;  d = (v - S k) * beta ;  S <- S + d k^T

así que para rehacer un token alcanza con guardar ``(k normalizada, v, g, beta)``,
~8 KB por token por capa, en vez del estado. Entonces:

* forward spec: lee el estado del slot de la columna 0, **reproduce
  num_accepted-1 filas de la cinta** del paso anterior, procesa los K+1 tokens,
  escribe el estado tras el token 0 (siempre aceptado) y la cinta de 1..K;
* las copias de ``align`` (``mamba_utils``) que upstream hace leyendo
  ``estado[columna + bias]`` se reemplazan por ``estado[columna]`` + ``bias``
  filas de cinta. La contabilidad de upstream (``num_accepted``, columnas,
  bordes cacheados) queda intacta: el estado que se materializa es el mismo.

La cinta NO entra en el relleno de la página (con spec decode el estado conv es
de kernel-1+K columnas y el relleno queda en 4.096 B), así que vive en un buffer
por capa indexado por un SLOT por request: ``[slots, K, fila]`` en fp16, ~13 MB
por rank con max-num-seqs 10.

Validado aislado (tests/proto/gdn_cinta.py): error relativo 1,9e-4 contra
upstream en 40 pasos con aceptaciones al azar (redondeo fp16).
"""

from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("genesis.pn122")

_TRUTHY = ("1", "true", "yes", "on")
_ACTIVO = os.environ.get("GENESIS_ENABLE_PN122_GDN_CINTA", "0").strip().lower() in _TRUTHY


def activo() -> bool:
    return _ACTIVO


_DEBUG = os.environ.get("GENESIS_PN122_DEBUG", "0").strip().lower() in _TRUTHY
_debug_n = 0


_SYNC_FILE = "/dev/shm/pn122_sync"


_sync_cache = [0, 0.0]


def sync_bits() -> int:
    """Bitmask de sincronizaciones de diagnostico, leido en caliente de un archivo.

    Se relee como mucho una vez por segundo: leerlo en cada llamada frenaba la
    CPU lo suficiente como para ESCONDER la carrera que se esta buscando.
    """
    if not _DEBUG:
        return 0
    import time
    now = time.monotonic()
    if now - _sync_cache[1] > 1.0:
        _sync_cache[1] = now
        try:
            with open(_SYNC_FILE) as f:
                _sync_cache[0] = int(f.read().strip() or 0)
        except (OSError, ValueError):
            _sync_cache[0] = 0
    return _sync_cache[0]


def debug_builder(num_spec_decodes, nacc, slots, sidx, cu, seq_lens) -> None:
    """Diagnostico (sincroniza con la GPU: solo con GENESIS_PN122_DEBUG=1)."""
    global _debug_n
    bits = sync_bits()
    if bits & 2:
        torch.cuda.synchronize()
    if not bits & 8 or num_spec_decodes == 0 or _debug_n > 400:
        return
    _debug_n += 1
    n = num_spec_decodes
    log.warning("[PN122 dbg] n=%d acc=%s slot=%s sidx=%s cu=%s seq=%s", n,
                nacc[:n].tolist() if nacc is not None else None,
                slots[:n].tolist() if slots is not None else None,
                sidx[:n, 0].tolist() if sidx is not None else None,
                cu[: n + 1].tolist() if cu is not None else None,
                seq_lens[:n].tolist() if seq_lens is not None else None)


def num_speculative_blocks(vllm_config) -> int:
    """Lo que upstream pone en ``MambaSpec.num_speculative_blocks``.

    Hay que reproducir la cuenta de cada version, porque con la cinta apagada este valor
    TIENE que ser exactamente el de upstream:

    * v0.27.1: ``speculative_config.num_speculative_tokens if speculative_config else 0``
    * v0.29.0: ``0 if cache_config.use_kda_recoverssm else num_speculative_tokens`` — el
      atajo subio a ``vllm_config`` y aparecio la rama de RecoverSSM, que verifica la
      ventana entera contra un solo checkpoint y por eso nunca escribe los slots por
      token de draft.
    """
    k = getattr(vllm_config, "num_speculative_tokens", None)
    if k is None:                                   # v0.27.1 y anteriores
        sc = vllm_config.speculative_config
        k = sc.num_speculative_tokens if sc else 0
    if getattr(vllm_config.cache_config, "use_kda_recoverssm", False):
        k = 0                                       # v0.29.0: RecoverSSM no usa esos slots
    # Diagnostico: GENESIS_PN122_SIN_LIBERAR=1 conserva los bloques especulativos
    # (cinta y kernel activos, sin ahorro) para aislar el efecto de sacarlos.
    if os.environ.get("GENESIS_PN122_SIN_LIBERAR", "0") == "1":
        return k
    return 0 if _ACTIVO else k


# ─────────────────────────────── slots ────────────────────────────────

_slot_de: dict[str, int] = {}
_libres: list[int] = []
_slots_cpu: torch.Tensor | None = None
_slots_gpu: torch.Tensor | None = None
_n_slots = 0
_ultimos_vals: list = []


def _init_slots(max_reqs: int, device) -> None:
    global _slots_cpu, _slots_gpu, _n_slots, _libres
    if _slots_gpu is not None:
        return
    # Slot 0 es de relleno (filas de padding de CUDA graph). El resto, holgado:
    # un slot se libera recien cuando el request termina o es preemptado.
    _n_slots = 2 * max_reqs + 2
    _libres = list(range(_n_slots - 1, 0, -1))
    # Filas: el builder indexa con la mascara spec del batch, que puede venir
    # rellenada para CUDA graph mas alla de max_reqs.
    filas = 4 * max_reqs + 8
    _slots_cpu = torch.zeros(filas, dtype=torch.int32, pin_memory=True)
    _slots_gpu = torch.zeros(filas, dtype=torch.int32, device=device)


def n_slots() -> int:
    return _n_slots


def slots_gpu() -> torch.Tensor:
    assert _slots_gpu is not None
    return _slots_gpu


def actualizar_slots(scheduler_output, req_ids: list[str], max_reqs: int, device,
                     vivos) -> None:
    """Corre en ``preprocess_mamba`` (antes de la metadata y del forward).

    Deja en ``slots_gpu()[i]`` el slot de cinta del request de la fila ``i`` del
    batch, que es el orden que usan el builder de GDN y los kernels de copia.

    ``vivos`` es ``GPUModelRunner.requests``. NO alcanza con
    ``finished_req_ids``: un request que termina en un paso sin tokens agendados
    nunca pasa por aca (``execute_model`` sale antes) y su slot se fugaba —
    medido: el engine murio con 21 slots ocupados tras 21 requests seriales.
    """
    _init_slots(max_reqs, device)
    pre = scheduler_output.preempted_req_ids or set()
    res = scheduler_output.scheduled_cached_reqs.resumed_req_ids
    muertos = [rid for rid in _slot_de if rid not in vivos or rid in pre or rid in res]
    for rid in muertos:
        _libres.append(_slot_de.pop(rid))
    n = len(req_ids)
    vals = []
    for rid in req_ids:
        s = _slot_de.get(rid)
        if s is None:
            if not _libres:
                raise RuntimeError(
                    "[PN122] sin slots de cinta: %d requests vivos con %d slots"
                    % (len(_slot_de), _n_slots - 1))
            s = _libres.pop()
            _slot_de[rid] = s
            _sombra_nuevos.add(s)
        vals.append(s)
    global _ultimos_vals
    if n and vals != _ultimos_vals:
        _ultimos_vals = vals
        # Desde memoria NO pinned a proposito: con un buffer pinned reutilizado
        # y non_blocking, el paso siguiente podia pisar los valores antes de
        # que la GPU (atrasada, async scheduling) ejecutara la copia, y el
        # kernel reproducia la cinta de OTRO request.
        _slots_gpu[:n].copy_(torch.tensor(vals, dtype=torch.int32), non_blocking=True)
    if sync_bits() & 1:
        torch.cuda.synchronize()


# ─────────────────────────────── cinta por capa ────────────────────────────────

def dims(layer) -> tuple[int, int, int, int]:
    tp = layer.tp_size
    return (layer.num_k_heads // tp, layer.num_v_heads // tp,
            layer.head_k_dim, layer.head_v_dim)


def fila(layer) -> int:
    H, HV, K, V = dims(layer)
    return H * K + HV * V + 2 * HV


def enlazar(layer, device) -> None:
    """Crea la cinta de la capa GDN. Se llama desde ``bind_kv_cache``."""
    if not hasattr(layer, "num_v_heads") or getattr(layer, "num_spec", 0) <= 0:
        return
    if getattr(layer, "_g122_cinta", None) is not None:
        return
    try:
        from vllm.config import get_current_vllm_config
        max_reqs = int(get_current_vllm_config().scheduler_config.max_num_seqs)
    except Exception:
        max_reqs = 32
    _init_slots(max_reqs, device)
    log.warning("[PN122] cinta de %s: %d slots x %d tokens x %d", getattr(layer, "prefix", "?"),
                _n_slots, layer.num_spec, fila(layer))
    layer._g122_cinta = torch.zeros(
        (_n_slots, layer.num_spec, fila(layer)), dtype=torch.float16, device=device)


# ─────────────────────────────── kernels ────────────────────────────────

from vllm.triton_utils import tl, triton  # noqa: E402


@triton.jit(do_not_specialize=["N"])
def _k_spec(A_log, a, b, dt_bias, beta_sp, threshold, q, k, v, o, h, stride_h, cu, sidx,
            nacc, slots, cinta, scale, N,
            H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
            BK: tl.constexpr, BV: tl.constexpr, TM: tl.constexpr, ROW: tl.constexpr,
            IS_L2: tl.constexpr):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos = tl.load(cu + i_n).to(tl.int64)
    eos = tl.load(cu + i_n + 1).to(tl.int64)
    T = eos - bos
    if T == 0:
        return
    s = tl.load(sidx + i_n).to(tl.int64)
    if s <= 0:
        return
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mk = o_k < K
    mv = o_v < V
    mh = mv[:, None] & mk[None, :]
    Al = tl.load(A_log + i_hv).to(tl.float32)
    db = tl.load(dt_bias + i_hv).to(tl.float32)
    p_h = h + s * stride_h + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(p_h, mask=mh, other=0).to(tl.float32)

    # 1) reproducir las filas aceptadas del paso anterior
    r = tl.load(nacc + i_n).to(tl.int64) - 1
    slot = tl.load(slots + i_n).to(tl.int64)
    for j in range(0, r):
        row = cinta + (slot * TM + j) * ROW
        rk = tl.load(row + i_h * K + o_k, mask=mk, other=0).to(tl.float32)
        rv = tl.load(row + H * K + i_hv * V + o_v, mask=mv, other=0).to(tl.float32)
        rg = tl.load(row + H * K + HV * V + i_hv).to(tl.float32)
        rb = tl.load(row + H * K + HV * V + HV + i_hv).to(tl.float32)
        b_h *= tl.exp(rg)
        b_d = (rv - tl.sum(b_h * rk[None, :], 1)) * rb
        b_h += b_d[:, None] * rk[None, :]

    # 2) tokens del paso actual, igual que upstream
    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_a = a + bos * HV + i_hv
    p_b = b + bos * HV + i_hv
    p_o = o + (bos * HV + i_hv) * V + o_v
    for t in range(0, T):
        b_q = tl.load(p_q, mask=mk, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mv, other=0).to(tl.float32)
        x = tl.load(p_a).to(tl.float32) + db
        sp = tl.where(beta_sp * x <= threshold, (1 / beta_sp) * tl.log(1 + tl.exp(beta_sp * x)), x)
        b_g = -tl.exp(Al) * sp
        b_beta = tl.sigmoid(tl.load(p_b).to(tl.float32))
        if IS_L2:
            b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        b_h *= tl.exp(b_g)
        b_d = (b_v - tl.sum(b_h * b_k[None, :], 1)) * b_beta
        b_h += b_d[:, None] * b_k[None, :]
        tl.store(p_o, tl.sum(b_h * b_q[None, :], 1).to(p_o.dtype.element_ty), mask=mv)
        if t == 0:  # la unica escritura del estado completo
            tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mh)
        p_q += H * K
        p_k += H * K
        p_v += HV * V
        p_a += HV
        p_b += HV
        p_o += HV * V


@triton.jit(do_not_specialize=["N"])
def _k_escribir(A_log, a, b, dt_bias, beta_sp, threshold, k, v, cu, sidx, slots, cinta, N,
                H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                BK: tl.constexpr, BHV: tl.constexpr, BH: tl.constexpr,
                TM: tl.constexpr, ROW: tl.constexpr, IS_L2: tl.constexpr):
    """Filas de cinta de los tokens 1..T-1. Lanzamiento APARTE de ``_k_spec``:
    adentro, los programas que comparten cabeza k (y los escalares por cabeza v)
    pisaban lo que otro programa todavia estaba reproduciendo."""
    i_n = tl.program_id(0)
    bos = tl.load(cu + i_n).to(tl.int64)
    eos = tl.load(cu + i_n + 1).to(tl.int64)
    T = eos - bos
    s = tl.load(sidx + i_n).to(tl.int64)
    if s <= 0 or T <= 1:
        return
    slot = tl.load(slots + i_n).to(tl.int64)
    ok = tl.arange(0, BK)
    mk = ok < K
    ohv = tl.arange(0, BHV)
    mhv = ohv < HV * V
    oh = tl.arange(0, BH)
    mh = oh < HV
    Al = tl.load(A_log + oh, mask=mh, other=0).to(tl.float32)
    db = tl.load(dt_bias + oh, mask=mh, other=0).to(tl.float32)
    for t in range(1, T):
        src = bos + t
        row = cinta + (slot * TM + t - 1) * ROW
        for hh in range(0, H):
            kk = tl.load(k + (src * H + hh) * K + ok, mask=mk, other=0).to(tl.float32)
            if IS_L2:
                kk = kk * tl.rsqrt(tl.sum(kk * kk) + 1e-6)
            tl.store(row + hh * K + ok, kk.to(tl.float16), mask=mk)
        tl.store(row + H * K + ohv,
                 tl.load(v + src * HV * V + ohv, mask=mhv, other=0).to(tl.float16), mask=mhv)
        x = tl.load(a + src * HV + oh, mask=mh, other=0).to(tl.float32) + db
        sp = tl.where(beta_sp * x <= threshold, (1 / beta_sp) * tl.log(1 + tl.exp(beta_sp * x)), x)
        tl.store(row + H * K + HV * V + oh, (-tl.exp(Al) * sp).to(tl.float16), mask=mh)
        bb = tl.sigmoid(tl.load(b + src * HV + oh, mask=mh, other=0).to(tl.float32))
        tl.store(row + H * K + HV * V + HV + oh, bb.to(tl.float16), mask=mh)


def spec_update(layer, A_log, a, b, dt_bias, q, k, v, ssm_state, cu_seqlens,
                spec_state_indices, num_accepted_tokens, slots):
    """Reemplazo de ``fused_sigmoid_gating_delta_rule_update`` en el camino spec.

    Devuelve lo mismo que upstream: ``(o [1, T, HV, V], ssm_state)``.
    """
    _, Ttot, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    N = cu_seqlens.shape[0] - 1
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    cinta = layer._g122_cinta
    TM, ROW = cinta.shape[1], cinta.shape[2]
    sidx = spec_state_indices[:, 0].contiguous() if spec_state_indices.ndim == 2 \
        else spec_state_indices
    nacc = num_accepted_tokens
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    a, b = a.contiguous(), b.contiguous()
    _bits = sync_bits()
    if _bits & 64:
        torch.cuda.synchronize()
    _pref = getattr(layer, "prefix", "")
    sombra = (_bits & 32) and any(f".layers.{i}." in _pref for i in (0, 1, 2, 44))
    if sombra:
        _sombra_antes(layer, ssm_state, sidx, nacc, slots, N)
    o = q.new_empty(1, Ttot, HV, V)
    _k_spec[(triton.cdiv(V, BV), N * HV)](
        A_log, a, b, dt_bias, 1.0, 20.0, q, k, v, o, ssm_state, ssm_state.stride(0),
        cu_seqlens, sidx, nacc, slots, cinta, K ** -0.5, N,
        H=H, HV=HV, K=K, V=V, BK=BK, BV=BV, TM=TM, ROW=ROW, IS_L2=True,
        num_warps=4, num_stages=3)
    if _bits & 128:
        torch.cuda.synchronize()
    _k_escribir[(N,)](
        A_log, a, b, dt_bias, 1.0, 20.0, k, v, cu_seqlens, sidx, slots, cinta, N,
        H=H, HV=HV, K=K, V=V, BK=BK, BHV=triton.next_power_of_2(HV * V),
        BH=triton.next_power_of_2(HV), TM=TM, ROW=ROW, IS_L2=True,
        num_warps=4, num_stages=3)
    if _bits & 256:
        torch.cuda.synchronize()
    if sombra:
        _sombra_despues(layer, A_log, a, b, dt_bias, q, k, v, cu_seqlens, sidx, nacc, slots,
                        N, o, ssm_state)
    return o, ssm_state


# ─────────────── verificacion en sombra (diagnostico, capa 0, eager) ───────────────
_sombra: dict = {}
_sombra_nuevos: set = set()


def _sombra_antes(layer, ssm_state, sidx, nacc, slots, N):
    """Estado upstream: K+1 columnas por slot. La columna 0 de la sombra se toma
    del estado real SOLO en el primer paso spec del request (estado de prefill);
    despues la sombra evoluciona sola con el kernel de upstream y se compara.

    OJO CON LEER ``estado(antes)``: me hizo sacar una conclusion falsa (2026-09-19).
    Compara ``h[1 + slot*K1 + acc - 1]`` — el estado de UPSTREAM despues de acc-1 tokens —
    contra ``ssm_state[s]`` — el de PN122 despues del token 0. Esos dos solo coinciden cuando
    ``acc == 1``: para acc>1 TIENEN que diferir, porque PN122 recupera la diferencia
    reproduciendo la cinta, que es exactamente lo que el parche hace. Las muestras con acc>1
    no se pueden leer como error.

    Y aun con acc==1 el numero mezcla el error del paso con la DERIVA ACUMULADA: la sombra
    evoluciona sola desde que se siembra, asi que cualquier paso que el hook no vea (prefill,
    decode sin spec) la deja atras sin que la cinta tenga la culpa.

    Para decidir si PN122 esta bien NO alcanza ni esto ni comparar texto greedy: un error de
    1e-4 alcanza para dar vuelta un argmax y hacer divergir el texto sin que haya nada roto.
    Lo que si discrimina es la TASA DE ACEPTACION del spec decode, que se derrumbaria si el
    estado estuviera corrupto. Medido 2026-09-19 con DFlash2 K=8: 5,37 a 1k y 5,17 a 50k con
    PN122, contra 5,15-5,69 y 4,64-5,46 sin el. Dentro del rango: no hay corrupcion.
    """
    K1 = layer.num_spec + 1
    key = layer.prefix
    if key not in _sombra:
        _sombra[key] = torch.zeros((_n_slots * K1 + 1, *ssm_state.shape[1:]),
                                   dtype=ssm_state.dtype, device=ssm_state.device)
        _sombra[key + "nuevos"] = set(range(_n_slots))
    h = _sombra[key]
    nuevos = _sombra[key + "nuevos"]
    torch.cuda.synchronize()
    difs = []
    for i in range(N):
        s = int(sidx[i]); sl = int(slots[i]); acc = int(nacc[i])
        if s <= 0:
            continue
        if sl in _sombra_nuevos:
            for kk in list(_sombra):
                if kk.endswith("nuevos"):
                    _sombra[kk].add(sl)
            _sombra_nuevos.discard(sl)
        if sl in nuevos:
            nuevos.discard(sl)
            h[1 + sl * K1] = ssm_state[s]
            difs.append("init")
        else:
            esperado = h[1 + sl * K1 + acc - 1].float()
            real0 = ssm_state[s].float()
            difs.append(round(((esperado - real0).norm() / (esperado.norm() + 1e-9)).item(), 5))
    _sombra[key + "difs"] = difs


def _sombra_despues(layer, A_log, a, b, dt_bias, q, k, v, cu, sidx, nacc, slots, N, o, ssm_state):
    from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update as ref)
    K1 = layer.num_spec + 1
    h = _sombra[layer.prefix]
    cols = torch.zeros((N, K1), dtype=torch.int32, device=q.device)
    for i in range(N):
        sl = int(slots[i])
        cols[i] = torch.arange(1 + sl * K1, 1 + sl * K1 + K1, dtype=torch.int32)
        if int(sidx[i]) <= 0:
            cols[i] = 0
    o_ref, _ = ref(A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=q, k=k, v=v, initial_state=h,
                   inplace_final_state=True, cu_seqlens=cu, ssm_state_indices=cols,
                   num_accepted_tokens=nacc[:N].contiguous(), use_qk_l2norm_in_kernel=True)
    torch.cuda.synchronize()
    T = int(cu[N])
    err = ((o_ref[:, :T].float() - o[:, :T].float()).norm() / (o_ref[:, :T].float().norm() + 1e-9)).item()
    log.warning("[PN122 sombra] %s N=%d acc=%s err_salida=%.2e estado(antes)=%s",
                layer.prefix.split("model.")[-1], N, nacc[:N].tolist(), err,
                _sombra.get(layer.prefix + "difs"))


# ───────────────────── materializacion para las copias align ─────────────────────

@triton.jit(do_not_specialize=["num_reqs"])
def _k_materializar(MODO_POST: tl.constexpr, nacc, state_idx, nsched, ncomp, ndraft,
                    src_col_p, bias_p, bt_ptrs, bt_stride: tl.int64, ssm_addrs, ssm_strides,
                    grupos, cinta_addrs, slots, num_reqs, block_size: tl.constexpr,
                    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                    BK: tl.constexpr, BV: tl.constexpr, TM: tl.constexpr, ROW: tl.constexpr):
    """Escribe en ``dst`` el estado que upstream leeria de ``columna src + bias``:
    ``estado[src]`` + ``bias`` filas de cinta. Solo casos con bias > 0; el resto
    (conv, bias 0) lo sigue copiando el kernel de upstream."""
    req = tl.program_id(0)
    lay = tl.program_id(1)
    if req >= num_reqs:
        return
    if MODO_POST:
        acc = tl.load(nacc + req)
        src_col = tl.load(state_idx + req)
        running = tl.load(ncomp + req) + tl.load(nsched + req) - tl.load(ndraft + req)
        nuevo = running + acc - 1
        alineado = (nuevo // block_size) * block_size
        if alineado < running:
            return
        bias = alineado - running
        dst_col = alineado // block_size - 1
    else:
        src_col = tl.load(src_col_p + req)
        dst_col = tl.load(state_idx + req)
        if src_col < 0 or src_col == dst_col:
            return
        bias = tl.load(bias_p + req)
    if bias <= 0 or src_col < 0 or dst_col < 0:
        return
    g = tl.load(grupos + lay).to(tl.int64)
    bt = tl.load(bt_ptrs + g).to(tl.pointer_type(tl.int32)) + req * bt_stride
    sb = tl.load(bt + src_col).to(tl.int64)
    dbk = tl.load(bt + dst_col).to(tl.int64)
    if sb <= 0 or dbk <= 0:
        return
    base = tl.load(ssm_addrs + lay)
    stride = tl.load(ssm_strides + lay)
    src = (base + sb * stride).to(tl.pointer_type(tl.float16))
    dst = (base + dbk * stride).to(tl.pointer_type(tl.float16))
    slot = tl.load(slots + req).to(tl.int64)
    tape = tl.load(cinta_addrs + lay).to(tl.pointer_type(tl.float16)) + slot * TM * ROW
    o_k = tl.arange(0, BK)
    mk = o_k < K
    for hv in range(0, HV):
        ih = hv // (HV // H)
        for c in range(0, V, BV):
            o_v = c + tl.arange(0, BV)
            mv = o_v < V
            mh = mv[:, None] & mk[None, :]
            b_h = tl.load(src + hv * V * K + o_v[:, None] * K + o_k[None, :], mask=mh,
                          other=0).to(tl.float32)
            for j in range(0, bias):
                row = tape + j * ROW
                rk = tl.load(row + ih * K + o_k, mask=mk, other=0).to(tl.float32)
                rv = tl.load(row + H * K + hv * V + o_v, mask=mv, other=0).to(tl.float32)
                rg = tl.load(row + H * K + HV * V + hv).to(tl.float32)
                rb = tl.load(row + H * K + HV * V + HV + hv).to(tl.float32)
                b_h *= tl.exp(rg)
                b_d = (rv - tl.sum(b_h * rk[None, :], 1)) * rb
                b_h += b_d[:, None] * rk[None, :]
            tl.store(dst + hv * V * K + o_v[:, None] * K + o_k[None, :],
                     b_h.to(tl.float16), mask=mh)


class _Meta:
    capas: list = []
    ssm_addrs = None
    ssm_strides = None
    grupos = None
    cinta_addrs = None
    dims = None


_meta = _Meta()


def _init_meta(ctx, kv_cache_config, forward_context) -> bool:
    if _meta.ssm_addrs is not None:
        return True
    addrs, strides, grupos, cintas, capas = [], [], [], [], []
    for g_local, gid in enumerate(ctx.mamba_group_ids):
        for name in kv_cache_config.kv_cache_groups[gid].layer_names:
            layer = forward_context[name]
            cinta = getattr(layer, "_g122_cinta", None)
            if cinta is None:
                return False
            ssm = layer.kv_cache[1]
            assert ssm.dtype == torch.float16, "PN122 asume ssm fp16"
            addrs.append(ssm.data_ptr())
            strides.append(ssm.stride(0) * ssm.element_size())
            grupos.append(g_local)
            cintas.append(cinta.data_ptr())
            capas.append(layer)
    dev = capas[0]._g122_cinta.device
    _meta.ssm_addrs = torch.tensor(addrs, dtype=torch.int64, device=dev)
    _meta.ssm_strides = torch.tensor(strides, dtype=torch.int64, device=dev)
    _meta.grupos = torch.tensor(grupos, dtype=torch.int32, device=dev)
    _meta.cinta_addrs = torch.tensor(cintas, dtype=torch.int64, device=dev)
    _meta.capas = capas
    l0 = capas[0]
    _meta.dims = (*dims(l0), l0.num_spec, fila(l0))
    return True


def _lanzar(modo_post, ctx, num_reqs, nacc, state_idx, nsched, ncomp, ndraft, src_col, bias):
    H, HV, K, V, TM, ROW = _meta.dims
    _k_materializar[(num_reqs, len(_meta.capas))](
        modo_post, nacc, state_idx, nsched, ncomp, ndraft, src_col, bias,
        ctx.block_table_ptrs, ctx.block_table_stride_req, _meta.ssm_addrs,
        _meta.ssm_strides, _meta.grupos, _meta.cinta_addrs, slots_gpu(), num_reqs,
        block_size=ctx.block_size, H=H, HV=HV, K=K, V=V,
        BK=triton.next_power_of_2(K), BV=min(triton.next_power_of_2(V), 32),
        TM=TM, ROW=ROW, num_warps=4, num_stages=3)


def materializar_post(ctx, kv_cache_config, forward_context, num_reqs, nacc, state_idx_buf,
                      nsched_buf, ncomp_buf, ndraft_buf) -> None:
    """Recibe los CpuGpuBuffer de upstream: la decision de si ALGUN request puede
    cruzar un borde con bias > 0 se toma en CPU con los mismos valores que se
    subieron a la GPU, y si ninguno puede, no se lanza el kernel (ahorra un
    lanzamiento de Triton por paso de decode)."""
    bits = sync_bits()
    if bits & 4:  # diagnostico: saltear la materializacion post
        return
    if num_reqs == 0:
        return
    import numpy as np
    comp = ncomp_buf.np[:num_reqs].astype(np.int64)
    sched = nsched_buf.np[:num_reqs].astype(np.int64)
    draft = ndraft_buf.np[:num_reqs].astype(np.int64)
    running = comp + sched - draft
    bs = ctx.block_size
    # bias = alineado - running > 0 exige un borde en (running, running + draft].
    if not ((((running + draft) // bs) * bs) > running).any():
        return
    if not _init_meta(ctx, kv_cache_config, forward_context):
        return
    _lanzar(True, ctx, num_reqs, nacc, state_idx_buf.gpu, nsched_buf.gpu, ncomp_buf.gpu,
            ndraft_buf.gpu, nacc, nacc)


def materializar_pre(ctx, kv_cache_config, forward_context, num_reqs, state_idx_buf,
                     src_col_buf, bias_buf) -> None:
    """Idem: solo se lanza si algun request migra de bloque con bias > 0."""
    if num_reqs == 0:
        return
    src = src_col_buf.np[:num_reqs]
    bias = bias_buf.np[:num_reqs]
    if not ((src >= 0) & (bias > 0)).any():
        return
    if not _init_meta(ctx, kv_cache_config, forward_context):
        return
    g = state_idx_buf.gpu
    _lanzar(False, ctx, num_reqs, g, g, g, g, g, src_col_buf.gpu, bias_buf.gpu)


__all__ = ["activo", "num_speculative_blocks", "actualizar_slots", "enlazar",
           "spec_update", "materializar_post", "materializar_pre", "slots_gpu"]
