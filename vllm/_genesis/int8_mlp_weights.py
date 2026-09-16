# SPDX-License-Identifier: Apache-2.0
"""Conversion de los pesos MLP de AWQ int4-por-grupo a int8 per-canal.

Para que
--------
El bloque `gate_up + SiLU` del MLP es el **43% del chunk de prefill** y hoy
corre en fp16 por Marlin. Medido en esta maquina, misma forma (M=2048 K=5120
N=8704x2), con la GPU libre:

    fp16 denso cuBLAS + SiLU     5,996 ms    60,9 TOPS
    Marlin W4A16 + SiLU          6,330 ms    57,7 TOPS   <- lo que corre hoy
    cutlass INT8 + SiLU          3,591 ms   101,7 TOPS   1,76x
    SK-12 W8A8 fusionado         3,775 ms    96,7 TOPS   1,68x
    SK-13 W4A8 fusionado         4,328 ms    84,3 TOPS   1,46x

Las dos opciones INT8 piden pesos int8, que este checkpoint no tiene.

Por que REEMPLAZAR y no duplicar
--------------------------------
Medido del propio checkpoint:

    MLP hoy (int4 + escalas + zp)   9,277 GiB  ->  4,639 por GPU
    MLP en int8 per-canal          15,938 GiB  ->  7,969 por GPU
                                    delta al REEMPLAZAR: +3,330 GiB

Si el int4 se libera despues de convertir, el costo es la diferencia, no la
suma. Con 1,14 GiB libres hay que sacarle 2,19 GiB al KV cache:
701.376 -> ~566.000 tokens de contexto, contra un working set de 380.000.
Entra.

Y hay un efecto util: vLLM dimensiona el KV cache DESPUES de cargar los pesos,
midiendo la VRAM libre. Si la conversion ocurre antes de ese profiling, el KV
se achica solo y no hay que tocar --gpu-memory-utilization a mano.

Precision
---------
Pasar de 4 bits por grupos de 128 a 8 bits per-canal SOLO funciona si las
escalas de grupo de un canal no varian mucho. Medido en este checkpoint:

    capa            rango de escalas   err int8/canal   err int8/grupo
    L0  gate_proj        1,7x              0,905%          0,656%
    L63 down_proj        1,9x              1,308%          0,680%

1,7-2,0x de rango es angosto, asi que el error extra es ~1% sobre lo que el
int4 ya arrastra (3-5% tipico). `medir_error()` lo recalcula por capa para
auditar antes de confiar.

Sobre el cache en disco
-----------------------
Esta implementado pero viene APAGADO, y conviene que siga asi: el int8 pesa el
DOBLE que el int4 en disco (16 GiB contra 9), asi que leer el cache es mas
lento que convertir. La conversion son ~5 pasadas de memoria sobre 89M
elementos por proyeccion, ~3 ms en GPU, ~0,4 s para las 128 del modelo.
Se deja por si en otra maquina el disco es mas rapido que la GPU.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pathlib

import torch

log = logging.getLogger("genesis.int8_mlp_weights")

GROUP = 128


def _desempaquetar(valor: torch.Tensor, forma: torch.Size, dim: int) -> torch.Tensor:
    """int32 con 8 nibbles -> int8 con valores 0..15.

    Mismo orden de bits que `compressed_tensors.unpack_from_int32`:
    `salida[..., i::8] = (valor >> 4i) & 0xF`, y despues `- 8` para pasar de
    sin signo a con signo (0..15 -> -8..7). Se reimplementa en vez de
    importarlo para no depender del paquete en tiempo de carga del modelo, y
    `test_pn118_int8_mlp` verifica que coincidan bit a bit.

    El `- 8` es inocuo para la dequantizacion porque se le resta lo mismo a los
    pesos y al zero point, y se cancela en `(w4 - zp)`. Se aplica igual para no
    separarse del contrato de upstream: si alguien usa esta funcion para otra
    cosa que no sea esa resta, sin el offset daria valores corridos en 8.
    """
    if valor.dtype is not torch.int32:
        raise ValueError(f"esperaba int32, llego {valor.dtype}")
    pf = 8
    if dim == 1:
        out = torch.zeros((valor.shape[0], valor.shape[1] * pf),
                          device=valor.device, dtype=torch.int32)
        for i in range(pf):
            out[:, i::pf] = (valor >> (4 * i)) & 0xF
        return (out[:, :forma[1]] - 8).to(torch.int8)
    out = torch.zeros((valor.shape[0] * pf, valor.shape[1]),
                      device=valor.device, dtype=torch.int32)
    for i in range(pf):
        out[i::pf, :] = (valor >> (4 * i)) & 0xF
    return (out[:forma[0], :] - 8).to(torch.int8)


def dequantizar(weight_packed: torch.Tensor, weight_scale: torch.Tensor,
                weight_zero_point: torch.Tensor) -> torch.Tensor:
    """AWQ pack-quantized -> pesos densos fp32 [N, K].

    El zero point viene empaquetado a lo largo de N (no de K), que es facil de
    confundir: `weight_zero_point` es [N/8, G] mientras `weight_packed` es
    [N, K/8].
    """
    N, G = weight_scale.shape
    K = weight_packed.shape[1] * 8
    w4 = _desempaquetar(weight_packed, torch.Size((N, K)), dim=1).float()
    zp = _desempaquetar(weight_zero_point, torch.Size((N, G)), dim=0).float()
    rep = K // G
    return (w4 - zp.repeat_interleave(rep, dim=1)) * \
        weight_scale.float().repeat_interleave(rep, dim=1)


def a_int8_per_canal(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pesos densos [N, K] -> (int8 [N, K], escala fp32 [N]).

    Simetrico y per-canal de salida: es el formato que consumen tanto
    `cutlass_scaled_mm` como SK-12.
    """
    s = w.abs().amax(dim=1) / 127.0
    s = s.clamp_min(1e-12)
    w8 = torch.round(w / s[:, None]).clamp(-127, 127).to(torch.int8)
    return w8, s.float()


def convertir(weight_packed, weight_scale, weight_zero_point, filas=4096):
    """AWQ -> (int8 [N, K], escala [N]), por bloques de filas.

    Va por bloques para que el pico de memoria sea el bloque y no la capa
    entera: una gate_up de [17408, 5120] en fp32 son 356 MB de una sola vez.
    """
    N = weight_scale.shape[0]
    K = weight_packed.shape[1] * 8
    w8 = torch.empty((N, K), dtype=torch.int8, device=weight_packed.device)
    esc = torch.empty(N, dtype=torch.float32, device=weight_packed.device)
    for a in range(0, N, filas):
        b = min(a + filas, N)
        zp_a, zp_b = a // 8, (b + 7) // 8
        bloque = dequantizar(
            weight_packed[a:b], weight_scale[a:b],
            weight_zero_point[zp_a:zp_b])
        # El desempaque del zp redondea a multiplos de 8 filas; recortar.
        bloque = bloque[: b - a]
        w8[a:b], esc[a:b] = a_int8_per_canal(bloque)
        del bloque
    return w8, esc


def medir_error(weight_packed, weight_scale, weight_zero_point) -> dict:
    """Error relativo que introduce la conversion. Para auditar por capa."""
    w = dequantizar(weight_packed, weight_scale, weight_zero_point)
    w8, s = a_int8_per_canal(w)
    rec = w8.float() * s[:, None]
    den = w.abs().mean().clamp_min(1e-12)
    sc = weight_scale.float()
    return {
        "err_medio": float((rec - w).abs().mean() / den),
        "err_max": float((rec - w).abs().max() / den),
        "rango_escalas": float(
            (sc.amax(dim=1) / sc.amin(dim=1).clamp_min(1e-9)).median()),
    }


# ─────────────────────────── cache en disco (opt-in) ───────────────────────────

def _cache_activo() -> bool:
    return os.environ.get("GENESIS_INT8_CACHE", "1").strip().lower() in (
        "1", "true", "yes", "on")


def _dir_cache() -> pathlib.Path:
    d = pathlib.Path(os.environ.get(
        "GENESIS_INT8_CACHE_DIR",
        os.path.expanduser("~/.cache/genesis/int8_mlp")))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _clave(modelo: str, capa: str, rank: int, forma) -> str:
    h = hashlib.sha256(f"{modelo}|{capa}|{rank}|{tuple(forma)}".encode())
    return h.hexdigest()[:24]


def obtener(weight_packed, weight_scale, weight_zero_point,
            modelo: str = "", capa: str = "", rank: int = 0):
    """Convierte, usando el cache en disco si `GENESIS_INT8_CACHE=1`."""
    if not _cache_activo():
        return convertir(weight_packed, weight_scale, weight_zero_point)

    # La huella del contenido va en la clave como segunda red: aunque el rank
    # viniera mal, dos fragmentos de TP distintos no pueden compartir entrada.
    huella = (int(weight_packed.reshape(-1)[:65536].to(torch.int64).sum())
              ^ int(weight_scale.reshape(-1)[:4096].float().sum().item() * 1e6))
    k = _clave(modelo, f"{capa}|{huella}", rank, weight_packed.shape)
    f = _dir_cache() / f"{k}.pt"
    if f.is_file():
        try:
            d = torch.load(f, map_location=weight_packed.device)
            return d["w8"], d["escala"]
        except Exception as e:
            log.warning("[int8] cache ilegible %s: %s — reconvierto", f, e)
    w8, esc = convertir(weight_packed, weight_scale, weight_zero_point)
    try:
        tmp = f.with_suffix(".tmp")
        torch.save({"w8": w8.cpu(), "escala": esc.cpu()}, tmp)
        os.replace(tmp, f)
    except Exception as e:
        log.warning("[int8] no se pudo escribir el cache %s: %s", f, e)
    return w8, esc


__all__ = ["convertir", "dequantizar", "a_int8_per_canal", "medir_error",
           "obtener", "GROUP"]
