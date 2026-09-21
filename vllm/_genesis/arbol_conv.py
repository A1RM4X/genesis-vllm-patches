# SPDX-License-Identifier: Apache-2.0
"""Arbol de borrador — hito 4: la convolucion causal de GDN, por CAMINO.

La conv1d de GDN (ancho 4) mezcla cada token con los 3 anteriores. En un paso en arbol "los 3
anteriores" de un nodo no son los 3 tokens previos del lote sino sus 3 ANCESTROS mas cercanos
(y, si no alcanzan, el ancla y la historia del estado conv).

El kernel de upstream (``causal_conv1d_update``) no se toca, porque ademas de las salidas hace
algo que sigue sirviendo tal cual: escribir el estado ``[h-2, h-1, x0, x1, ..., xK]``. Entonces:

* ``salidas``: calcula aparte las salidas por camino. Se llama ANTES que upstream (que pisa
  ``x`` con sus salidas de cadena) y el resultado se copia encima despues.
* ``compactar``: despues de aceptar. El paso siguiente lee las columnas ``r, r+1, r+2`` del
  estado (``r`` = borradores aceptados) creyendo que ahi estan los 3 ultimos tokens aceptados.
  Con una cadena lo estan; con un arbol el j-esimo aceptado es el nodo ``p_j`` y vive en la
  columna ``2 + p_j``, asi que se copian a lo sumo 3 columnas. Nunca se lee una columna ya
  pisada: el origen de la columna ``r+i`` es ``2 + p_(r+i-2) >= r+i``, estrictamente creciente.

Todo en GPU, forma fija, sin sincronizar. Con la mascara de cadena ``salidas`` da lo mismo que
upstream y ``compactar`` no copia nada.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _k_salidas(x, stride_xt, out, stride_ot, cs, stride_cs_seq, stride_cs_dim, stride_cs_tok,
               w, stride_wd, stride_ww, cu, sidx, nacc, anc, DIM,
               BN: tl.constexpr, SILU: tl.constexpr):
    i_n, i_f = tl.program_id(0), tl.program_id(1)
    bos = tl.load(cu + i_n).to(tl.int64)
    T = tl.load(cu + i_n + 1).to(tl.int64) - bos
    s = tl.load(sidx + i_n).to(tl.int64)
    if T == 0 or s <= 0:
        return
    f = i_f * BN + tl.arange(0, BN)
    mf = f < DIM
    off = tl.load(nacc + i_n).to(tl.int64) - 1
    hist = cs + s * stride_cs_seq + f * stride_cs_dim + off * stride_cs_tok
    hA = tl.load(hist, mask=mf, other=0.0)                      # la mas vieja
    hB = tl.load(hist + stride_cs_tok, mask=mf, other=0.0)
    hC = tl.load(hist + 2 * stride_cs_tok, mask=mf, other=0.0)
    w0 = tl.load(w + f * stride_wd, mask=mf, other=0.0)
    w1 = tl.load(w + f * stride_wd + stride_ww, mask=mf, other=0.0)
    w2 = tl.load(w + f * stride_wd + 2 * stride_ww, mask=mf, other=0.0)
    w3 = tl.load(w + f * stride_wd + 3 * stride_ww, mask=mf, other=0.0)
    x0 = tl.load(x + bos * stride_xt + f, mask=mf, other=0.0)
    for t in range(0, T):
        xt = tl.load(x + (bos + t) * stride_xt + f, mask=mf, other=0.0)
        t1 = hC                                   # t1 = el anterior inmediato ... t3 = el mas lejano
        t2 = hB
        t3 = hA
        if t > 0:
            m = tl.load(anc + bos + t).to(tl.int32)
            cnt = 0
            for i in range(0, t - 1):             # ancestros de t, del mas cercano al mas lejano
                j = t - 1 - i
                if cnt < 3:
                    if ((m >> (j - 1)) & 1) != 0:
                        xj = tl.load(x + (bos + j) * stride_xt + f, mask=mf, other=0.0)
                        if cnt == 0:
                            t1 = xj
                        elif cnt == 1:
                            t2 = xj
                        else:
                            t3 = xj
                        cnt += 1
            if cnt == 0:                          # lo que falta sale de [x0, hC, hB]
                t1 = x0
                t2 = hC
                t3 = hB
            elif cnt == 1:
                t2 = x0
                t3 = hC
            elif cnt == 2:
                t3 = x0
        # Igual que upstream, para que la cadena de bit a bit lo mismo: cada producto en el
        # dtype de x (fp16) y la suma en fp32, de la columna mas vieja a la mas nueva.
        acc = tl.zeros((BN,), dtype=tl.float32)
        acc += t3 * w0
        acc += t2 * w1
        acc += t1 * w2
        acc += xt * w3
        if SILU:
            acc = acc / (1 + tl.exp(-acc))
        tl.store(out + (bos + t) * stride_ot + f, acc.to(out.dtype.element_ty), mask=mf)


def salidas(x, conv_state, weight, activation, conv_state_indices, num_accepted_tokens,
            query_start_loc, anc, out=None):
    """Mismos argumentos que ``causal_conv1d_update`` en el camino spec (``x`` [tokens, dim],
    ``conv_state`` [bloques, dim, state_len], ``weight`` [dim, 4], sin bias) mas ``anc``.
    Devuelve las salidas por camino en ``out`` (no toca ``x`` ni el estado)."""
    assert weight.shape[1] == 4 and x.stride(1) == 1
    if out is None:
        out = torch.empty_like(x)
    N = query_start_loc.shape[0] - 1
    dim = x.shape[1]
    BN = 1024
    conv_state_indices = conv_state_indices.contiguous()      # en vLLM llega como columna de un 2D
    _k_salidas[(N, triton.cdiv(dim, BN))](
        x, x.stride(0), out, out.stride(0), conv_state, conv_state.stride(0),
        conv_state.stride(1), conv_state.stride(2), weight, weight.stride(0), weight.stride(1),
        query_start_loc, conv_state_indices, num_accepted_tokens, anc, dim,
        BN=BN, SILU=activation in ("silu", "swish", True), num_warps=4)
    return out


@triton.jit(do_not_specialize=["num_reqs"])
def _k_compactar(conv_addrs, grupos, bloques, stride_bloq_g, nacc, camino, fila_camino, num_reqs,
                 stride_cs_seq, stride_cs_dim, stride_cs_tok, DIM,
                 TM: tl.constexpr, BN: tl.constexpr):
    req, lay = tl.program_id(0), tl.program_id(1)
    if req >= num_reqs:
        return
    r = tl.load(nacc + req).to(tl.int64) - 1
    if r <= 0:
        return
    g = tl.load(grupos + lay).to(tl.int64)
    blk = tl.load(bloques + g * stride_bloq_g + req).to(tl.int64)
    if blk <= 0:
        return
    fc = tl.load(fila_camino + req).to(tl.int64)
    base = tl.load(conv_addrs + lay).to(tl.pointer_type(tl.float16)) + blk * stride_cs_seq
    for i in range(0, 3):
        sq = r + i - 3                            # indice en el camino aceptado (0 = p_1)
        if sq >= 0:
            src = 3 + tl.load(camino + fc * TM + sq).to(tl.int64)
            dst = r + i
            if src != dst:
                for c in range(0, DIM, BN):
                    f = c + tl.arange(0, BN)
                    mf = f < DIM
                    val = tl.load(base + f * stride_cs_dim + src * stride_cs_tok, mask=mf, other=0.0)
                    tl.store(base + f * stride_cs_dim + dst * stride_cs_tok, val, mask=mf)


def compactar(conv_addrs, grupos, bloques, nacc, camino, fila_camino, num_reqs, strides, dim):
    """Despues de aceptar. ``conv_addrs`` [capas] int64 (``data_ptr`` del estado conv fp16 de cada
    capa, todas con los mismos ``strides`` = (bloque, dim, columna) en ELEMENTOS), ``grupos``
    [capas] -> fila de ``bloques`` [G, reqs] (bloque del estado de cada pedido), ``nacc`` [reqs]
    = aceptados del paso (ancla incluida), ``camino`` [slots, TM] filas de cinta del camino
    aceptado (nodo - 1) y ``fila_camino`` [reqs] = que fila de ``camino`` usa cada pedido."""
    _k_compactar[(num_reqs, conv_addrs.shape[0])](
        conv_addrs, grupos, bloques, bloques.stride(0), nacc, camino, fila_camino, num_reqs,
        strides[0], strides[1], strides[2], dim, TM=camino.shape[1], BN=1024, num_warps=4)


@triton.jit(do_not_specialize=["num_reqs"])
def _k_compactar_v2(conv_addrs, grupos, bt_ptrs, bt_stride: tl.int64, state_idx, idx_map, nacc,
                    camino, num_reqs, stride_cs_seq, stride_cs_dim, stride_cs_tok, DIM,
                    TM: tl.constexpr, BN: tl.constexpr):
    """Como ``_k_compactar``, con el direccionamiento del runner v2 (el mismo de
    ``gdn_cinta._k_materializar``): el bloque del estado sale de la block table del grupo, en la
    columna ``state_idx[slot de req-state]``; la fila de ``camino`` es ``slot + 1``; ``nacc`` viene
    por fila del lote (es ``num_sampled``, todavia sin volcar a ``num_accepted_tokens``)."""
    req, lay = tl.program_id(0), tl.program_id(1)
    if req >= num_reqs:
        return
    ri = tl.load(idx_map + req)
    if ri < 0:
        return
    r = tl.load(nacc + req).to(tl.int64) - 1
    if r <= 0:
        return
    col = tl.load(state_idx + ri)
    if col < 0:
        return
    g = tl.load(grupos + lay).to(tl.int64)
    bt = tl.load(bt_ptrs + g).to(tl.pointer_type(tl.int32)) + req * bt_stride
    blk = tl.load(bt + col).to(tl.int64)
    if blk <= 0:
        return
    base = tl.load(conv_addrs + lay).to(tl.pointer_type(tl.float16)) + blk * stride_cs_seq
    fc = (ri + 1).to(tl.int64)
    for i in range(0, 3):
        sq = r + i - 3
        if sq >= 0:
            src = 3 + tl.load(camino + fc * TM + sq).to(tl.int64)
            dst = r + i
            if src != dst:
                for c in range(0, DIM, BN):
                    f = c + tl.arange(0, BN)
                    mf = f < DIM
                    val = tl.load(base + f * stride_cs_dim + src * stride_cs_tok, mask=mf, other=0.0)
                    tl.store(base + f * stride_cs_dim + dst * stride_cs_tok, val, mask=mf)


_v2 = {}


def compactar_v2(ctx, capas, grupos, state_idx, idx_mapping, nacc, camino, num_reqs) -> None:
    """``capas`` = capas GDN en el orden de ``grupos``. El estado conv es ``capa.kv_cache[0]``:
    2 bytes por elemento (se copia crudo, asi que fp16 y bf16 valen igual), con el eje de
    columnas de largo ``ancho - 1 + K``."""
    if not capas or num_reqs == 0:
        return
    if not _v2:
        c0 = capas[0].kv_cache[0]
        assert c0.element_size() == 2 and c0.dim() == 3, "estado conv inesperado"
        sl = 3 + camino.shape[1]
        eje_tok = 1 if c0.shape[1] == sl else 2
        assert c0.shape[eje_tok] == sl, f"estado conv {tuple(c0.shape)}: no hay eje de {sl} columnas"
        eje_dim = 3 - eje_tok
        _v2["strides"] = (c0.stride(0), c0.stride(eje_dim), c0.stride(eje_tok))
        _v2["dim"] = c0.shape[eje_dim]
        _v2["addrs"] = torch.tensor([c.kv_cache[0].data_ptr() for c in capas], dtype=torch.int64,
                                    device=c0.device)
    s = _v2["strides"]
    _k_compactar_v2[(num_reqs, len(capas))](
        _v2["addrs"], grupos, ctx.block_table_ptrs, ctx.block_table_stride_req, state_idx,
        idx_mapping, nacc, camino, num_reqs, s[0], s[1], s[2], _v2["dim"],
        TM=camino.shape[1], BN=1024, num_warps=4)


__all__ = ["salidas", "compactar", "compactar_v2"]
