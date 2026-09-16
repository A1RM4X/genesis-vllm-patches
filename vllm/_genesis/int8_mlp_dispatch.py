# SPDX-License-Identifier: Apache-2.0
"""PN118 — runtime: convierte el MLP a INT8 en la carga y despacha el forward.

Este modulo lo llama el codigo que PN118 inyecta en
`compressed_tensors_wNa16.py`. Vive aparte para que lo inyectado sean dos
lineas y toda la logica quede aca, versionada y testeable.

Por que un text patch y no un monkeypatch
-----------------------------------------
vLLM carga el modelo en procesos WORKER separados (`Worker_TP0`, `Worker_TP1`),
y `apply_all` no corre ahi: se verifico que no hay una sola linea de Genesis
con prefijo `Worker_TP` en el log. Un monkeypatch aplicado en el proceso padre
no llega al worker, asi que no puede tocar la carga de pesos. Solo los text
patches funcionan para esto, porque modifican el archivo en disco y lo ve todo
proceso que lo importe.

(La version anterior de PN118 era un monkeypatch y por eso no convertia nada:
el parche se instalaba, el log lo confirmaba, y el modelo seguia en Marlin.)
"""

from __future__ import annotations

import logging
import os
import time

import torch

log = logging.getLogger("genesis.pn118")

ATTR = "_genesis_int8_mlp"
_TRUTHY = ("1", "true", "yes", "on")


def _flag(nombre: str, default: str = "0") -> bool:
    return os.environ.get(nombre, default).strip().lower() in _TRUTHY


def activo() -> bool:
    return _flag("GENESIS_ENABLE_PN118_INT8_MLP")


def _patrones() -> tuple[str, ...]:
    """Que capas convertir, por subcadena del nombre.

    El perfil de nsys mostro que Marlin W4A16 se lleva el 25,3% del prefill: son
    las proyecciones de atencion y de las capas GDN, que el default (`.mlp.`)
    saltea. Convertirlas TODAS (`GENESIS_PN118_ALL_LAYERS=1`) no entra en VRAM
    —el arranque muere con OOM, ~2 GiB de mas por placa— asi que hace falta
    poder elegir un subconjunto:

        GENESIS_PN118_CAPAS=".mlp.,self_attn"   mlp + atencion, sin GDN
        GENESIS_PN118_CAPAS="self_attn"         solo atencion

    Vacio o `ALL_LAYERS=1` convierte todo.
    """
    if _flag("GENESIS_PN118_ALL_LAYERS"):
        return ()
    crudo = os.environ.get("GENESIS_PN118_CAPAS", ".mlp.").strip()
    if not crudo:
        return ()
    return tuple(p for p in (x.strip() for x in crudo.split(",")) if p)


def _conservar_int4() -> bool:
    """Conservar los int4 es solo para depurar: hace que el modelo pague las
    dos copias y el parche no ahorre nada."""
    return _flag("GENESIS_PN118_KEEP_INT4")


class _Stats:
    convertidas = 0
    salteadas = 0
    fallidas = 0
    segundos = 0.0
    bytes_int4 = 0
    bytes_int8 = 0
    desde_cache = 0
    err_max = 0.0


STATS = _Stats()


def _nombre(layer) -> str:
    return str(getattr(layer, "prefix", "") or getattr(layer, "_genesis_nombre", ""))


def _aplica(layer) -> bool:
    pats = _patrones()
    if not pats:
        return True
    n = _nombre(layer)
    return any(p in n for p in pats)


def _rank_tp() -> int:
    """Rank de TP REAL de este worker.

    Antes se leia ``os.environ["RANK"]``, que los workers multiproc de vLLM NO
    tienen: los dos ranks usaban rank=0, la clave del cache de conversion
    coincidia (los fragmentos de TP tienen la misma forma) y un rank cargaba el
    MLP convertido del OTRO. Quien escribia primero dependia del arranque: el
    modelo respondia mal en ~1 de cada 2 arranques y bien en los otros.
    """
    try:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
        return int(get_tensor_model_parallel_rank())
    except Exception:
        import torch
        return int(torch.cuda.current_device())


def convertir_capa(layer) -> bool:
    """Convierte la capa a INT8 y libera el int4.

    :returns: True si la capa quedo en INT8 (el caller NO debe llamar al camino
        original, porque los tensores AWQ ya no estan). False si no aplica.
    """
    if not activo():
        return False
    if not _aplica(layer):
        STATS.salteadas += 1
        return False

    wp = getattr(layer, "weight_packed", None)
    ws = getattr(layer, "weight_scale", None)
    wz = getattr(layer, "weight_zero_point", None)
    if wp is None or ws is None or wz is None:
        return False

    t0 = time.perf_counter()
    try:
        from vllm._genesis import int8_mlp_weights as conv

        wp_d = getattr(wp, "data", wp)
        ws_d = getattr(ws, "data", ws)
        wz_d = getattr(wz, "data", wz)

        if _flag("GENESIS_PN118_AUDIT"):
            try:
                e = conv.medir_error(wp_d, ws_d, wz_d)
                STATS.err_max = max(STATS.err_max, e["err_medio"])
                log.info("[PN118] %s err=%.4f%% rango_escalas=%.2fx",
                         _nombre(layer), 100 * e["err_medio"], e["rango_escalas"])
            except Exception as e:
                log.warning("[PN118] auditoria fallo: %s", e)

        antes = conv._cache_hit_count() if hasattr(conv, "_cache_hit_count") else 0
        w8, esc = conv.obtener(
            wp_d, ws_d, wz_d,
            modelo=os.environ.get("GENESIS_PN118_MODEL_ID", ""),
            capa=_nombre(layer), rank=_rank_tp())

        STATS.bytes_int4 += (wp_d.numel() * wp_d.element_size()
                             + ws_d.numel() * ws_d.element_size()
                             + wz_d.numel() * wz_d.element_size())
        STATS.bytes_int8 += w8.numel() + esc.numel() * 4

        # cutlass_scaled_mm pide B de [K, N] COLUMN-major, o sea strides (1, K).
        # Con w8 de [N, K] contiguo (strides (K, 1)), `w8.t()` da exactamente
        # eso. Hacer `.t().contiguous().t()` aca seria el error opuesto: deja
        # w8 column-major y entonces `w8.t()` sale row-major y cutlass lo
        # rechaza en scaled_mm_entry.cu.
        w8 = w8.contiguous()
        aplicar_24_en_carga(_nombre(layer), w8)
        setattr(layer, ATTR, (w8, esc.float()))
        from vllm._genesis import wush_mlp
        wush_mlp.preparar(layer, _nombre(layer), w8, esc)
        # Solo las capas que simulan (FAKE_SOLO): down_proj tiene K=8704 por placa y
        # su QR de 8704x8704 era el OOM del arranque.
        if _FAKE_ROT == "densa" and ".mlp." in _nombre(layer) and _FAKE_SOLO in _nombre(layer):
            _rot_densa(w8.shape[1], w8.device)
        STATS.convertidas += 1
    except Exception as e:
        STATS.fallidas += 1
        log.error("[PN118] %s NO se pudo convertir: %s", _nombre(layer), e)
        return False
    finally:
        STATS.segundos += time.perf_counter() - t0

    if _conservar_int4():
        return False   # deja correr el camino original tambien

    for n in ("weight_packed", "weight_scale", "weight_zero_point",
              "weight_g_idx", "weight_shape"):
        if hasattr(layer, n):
            try:
                delattr(layer, n)
            except Exception:
                pass
    return True


# ── Diagnostico de calidad de SK-14 (W4A4): cuantizacion SIMULADA ──────────────
# GENESIS_PN118_FAKE = "w4a4" | "w4" | "a4" | "" (apagado). Solo capas del MLP.
# Reproduce la aritmetica que haria el kernel s4 m16n8k64: int4 simetrico con una
# escala por grupo de GRUPO elementos a lo largo de K (64 = un paso de mma k64),
# para pesos (por fila) y activaciones (por token). Es lento (dequantiza en cada
# forward): sirve para medir calidad, no para produccion.
_FAKE = os.environ.get("GENESIS_PN118_FAKE", "").strip().lower()
# Subcadena del nombre para limitar la simulacion (p. ej. "gate_up" o "down_proj").
_FAKE_SOLO = os.environ.get("GENESIS_PN118_FAKE_SOLO", "").strip()
_FAKE_GRUPO = int(os.environ.get("GENESIS_PN118_FAKE_GRUPO", "64"))
# Granularidad por separado (0 = una escala por fila de W / por token de A) y
# bloque de la Hadamard (potencia de 2), para barrer el diseno del kernel s4.
_FAKE_GRUPO_W = int(os.environ.get("GENESIS_PN118_FAKE_GRUPO_W", str(_FAKE_GRUPO)))
_FAKE_GRUPO_A = int(os.environ.get("GENESIS_PN118_FAKE_GRUPO_A", str(_FAKE_GRUPO)))
_FAKE_ROT_BLOQUE = int(os.environ.get("GENESIS_PN118_FAKE_ROT_BLOQUE", str(_FAKE_GRUPO)))


def _q4g(t: torch.Tensor, g: int) -> torch.Tensor:
    sh = t.shape
    K = sh[-1]
    if g <= 0:
        g = K
    pad = (-K) % g
    y = torch.nn.functional.pad(t, (0, pad)) if pad else t
    y = y.reshape(*sh[:-1], -1, g)
    s = y.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 7.0
    y = (y / s).round().clamp(-7, 7) * s
    y = y.reshape(*sh[:-1], -1)
    return y[..., :K] if pad else y


_WUSH_CAPTURA = os.environ.get("GENESIS_PN118_WUSH", "").strip().lower().startswith("captura")
_FAKE_ROT = os.environ.get("GENESIS_PN118_FAKE_ROT", "").strip().lower()
# Dispersion simulada (P5/P6/P7): "w24" pesos 2:4 por magnitud a lo largo de K;
# "a816" activaciones 8:16 por token con puntaje |x|*norma de columna (Amber);
# "a24" activaciones 2:4 por token por magnitud. Combinables con coma.
_FAKE_DISP = os.environ.get("GENESIS_PN118_FAKE_DISP", "").strip().lower()


# Solo las variantes de activaciones necesitan el forward simulado; "w24" se aplica
# una vez en la carga sobre el int8 (la escala es por fila y positiva, asi que
# |w8| ordena igual que |W|) y el forward sigue por cutlass.
_FAKE_DISP_ACT = any(v in _FAKE_DISP for v in ("a816", "a24"))


def aplicar_24_en_carga(nombre: str, w8: torch.Tensor) -> None:
    if "w24" not in _FAKE_DISP or ".mlp." not in nombre:
        return
    # Por tandas de filas: topk sobre 17408x5120 de un tiro no entra en la VRAM libre.
    for lo in range(0, w8.shape[0], 1024):
        fila = w8[lo:lo + 1024]
        fila.mul_(_mascara_nm(fila.abs().float(), 2, 4).to(fila.dtype))


def _mascara_nm(score: torch.Tensor, n: int, m: int) -> torch.Tensor:
    """1 en las n posiciones de mayor puntaje de cada grupo de m (ultima dim)."""
    K = score.shape[-1]
    pad = (-K) % m
    s = torch.nn.functional.pad(score, (0, pad), value=-1.0) if pad else score
    s = s.reshape(*score.shape[:-1], -1, m)
    idx = s.topk(n, dim=-1).indices
    mk = torch.zeros_like(s).scatter_(-1, idx, 1.0)
    mk = mk.reshape(*score.shape[:-1], -1)
    return mk[..., :K] if pad else mk
_H64 = {}


def _rot_bloques(t: torch.Tensor, g: int) -> torch.Tensor:
    """Hadamard ortonormal por bloques de g a lo largo de la ultima dim (g potencia de 2).

    Es una rotacion ORTOGONAL exacta: (R x) . (R w) = x . w, asi que aplicada a
    activaciones y a las columnas del peso no cambia la cuenta, solo reparte los
    outliers dentro de cada grupo antes de cuantizar.
    """
    key = (g, t.device)
    if key not in _H64:
        # Construida en la GPU: una copia CPU->GPU durante la captura de CUDA
        # graphs aborta el arranque.
        H = torch.ones(1, 1, device=t.device, dtype=torch.float32)
        while H.shape[0] < g:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        _H64[key] = H / g ** 0.5
    H = _H64[key]
    K = t.shape[-1]
    pad = (-K) % g
    y = torch.nn.functional.pad(t, (0, pad)) if pad else t
    y = y.reshape(*t.shape[:-1], -1, g) @ H.t()
    y = y.reshape(*t.shape[:-1], -1)
    return y


_QD = {}


def _rot_densa(K: int, device) -> torch.Tensor:
    # Se construye en la CARGA (preparar_rot_densa): un torch.Generator dentro de
    # la region compilada aborta el arranque ("opaque object").
    key = (K, str(device))
    if key not in _QD:
        g = torch.Generator(device=device).manual_seed(1234)
        A = torch.randn(K, K, generator=g, device=device, dtype=torch.float32)
        Q, R = torch.linalg.qr(A)
        _QD[key] = (Q * torch.sign(torch.diagonal(R))[None, :]).contiguous()
        del A, Q, R
        torch.cuda.empty_cache()
    return _QD[key]


def _forward_simulado(w8, esc, x, bias, layer=None):
    forma = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1]).float()
    W = w8.float() * esc.float().view(-1, 1)
    wush = getattr(layer, "_g118_wush", None)
    if wush is not None:
        from vllm._genesis import wush_mlp
        x2 = wush_mlp.transformar(x2, wush[0])
        W = wush_mlp.transformar(W, wush[1])
    elif _FAKE_ROT == "densa":
        # Ortogonal densa de K x K (QR de una gaussiana con semilla fija, en la GPU):
        # reparte los outliers en TODO el vector para que alcance una escala por
        # token / por fila (el SK-14 rapido, sin volcado por grupo).
        Q = _rot_densa(x2.shape[-1], x2.device)
        x2 = x2 @ Q.t()
        W = W @ Q.t()
    elif _FAKE_ROT == "bloque":
        # La rotacion rellena K al multiplo de g; los rellenos son cero en x y en W,
        # asi que el producto no cambia.
        x2 = _rot_bloques(x2, _FAKE_ROT_BLOQUE)
        W = _rot_bloques(W, _FAKE_ROT_BLOQUE)
    if "a816" in _FAKE_DISP:
        x2 = x2 * _mascara_nm(x2.abs() * W.norm(dim=0).unsqueeze(0), 8, 16)
    if "a24" in _FAKE_DISP:
        x2 = x2 * _mascara_nm(x2.abs(), 2, 4)
    if "w4" in _FAKE:
        W = _q4g(W, _FAKE_GRUPO_W)
    if "a4" in _FAKE:
        x2 = _q4g(x2, _FAKE_GRUPO_A)
    out = x2 @ W.t()
    if bias is not None:
        out = out + bias.float()
    return out.to(x.dtype).reshape(*forma, out.shape[-1])


def forward_int8(layer, x, bias):
    """Forward por `cutlass_scaled_mm`. None si la capa no fue convertida.

    La activacion se cuantiza per-token dinamicamente: es el mismo esquema W8A8
    que usan los checkpoints INT8 oficiales.
    """
    estado = getattr(layer, ATTR, None)
    if estado is None:
        return None
    from vllm import _custom_ops as ops

    w8, esc = estado
    if (_FAKE or _FAKE_DISP_ACT) and ".mlp." in _nombre(layer) and _FAKE_SOLO in _nombre(layer):
        return _forward_simulado(w8, esc, x, bias, layer)
    if _WUSH_CAPTURA and ".mlp." in _nombre(layer):
        from vllm._genesis import wush_mlp
        wush_mlp.acumular(_nombre(layer), x.reshape(-1, x.shape[-1]))
    forma = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    x_i8, x_esc, _ = ops.scaled_int8_quant(x2, symmetric=True)
    out = ops.cutlass_scaled_mm(x_i8, w8.t(), x_esc, esc.view(1, -1),
                                x.dtype, bias)
    return out.reshape(*forma, out.shape[-1])


def informe() -> str:
    d = (STATS.bytes_int8 - STATS.bytes_int4) / (1024 ** 3)
    s = (f"{STATS.convertidas} capas MLP a INT8 en {STATS.segundos:.2f}s, "
         f"delta de pesos {d:+.3f} GiB")
    if STATS.err_max:
        s += f", err max {100 * STATS.err_max:.3f}%"
    if STATS.fallidas:
        s += f", {STATS.fallidas} FALLARON"
    return s


def stats() -> dict:
    return {"convertidas": STATS.convertidas, "salteadas": STATS.salteadas,
            "fallidas": STATS.fallidas, "segundos": round(STATS.segundos, 3),
            "delta_gib": round((STATS.bytes_int8 - STATS.bytes_int4) / 1024 ** 3, 3),
            "err_max": STATS.err_max}


__all__ = ["convertir_capa", "forward_int8", "activo", "stats", "informe", "ATTR"]
