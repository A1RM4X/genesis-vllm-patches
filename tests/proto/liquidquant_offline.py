#!/usr/bin/env python3
"""Cuanta precision cuesta el esquema de LiquidQuant, medido sobre los pesos REALES.

El kernel de LiquidQuant es 1,41x mas rapido en el desempaque (tests/proto/banco_dequant.cu)
porque mete la escala adentro del IMAD. El precio es que la escala tiene que ser **uint8** y el
producto ``Q_u4 * s_u8`` tiene que caber en 8 bits, mientras que PN130 hoy manda el peso int4
crudo al tensor core y aplica una escala **int16** sobre el acumulador int32.

Esto mide ese precio antes de escribir una linea de kernel, que es la parte cara. Si el error se
dispara, no hay nada que programar.

Los tres esquemas sobre el mismo peso:

  actual      w ~= (q4 - 8) * s_fp16              por grupo de 128, q4 en [0,15]
  liquid      w ~= ((q4 * s_u8 + a) ^ 0x80) * t   por grupo, todo entero hasta el final
  liquid+     igual pero con la escala de grupo en fp16 al final en vez de un t por canal

Uso: liquidquant_offline.py [n_tensores]
"""
from __future__ import annotations

import glob
import json
import os
import struct
import sys

import torch

G = 128
N_TENSORES = int(sys.argv[1]) if len(sys.argv) > 1 else 8


def cargar_pesos(n: int):
    """Devuelve (nombre, W fp16 [N,K]) de tensores GPTQ del checkpoint, ya dequantizados.

    El checkpoint ya esta en int4, asi que el fp16 "verdadero" no existe. Se usa como referencia
    el peso que el modelo realmente usa hoy — que es exactamente lo correcto para esta pregunta:
    cuanto se pierde AL PASAR del esquema de hoy al de LiquidQuant.
    """
    # El checkpoint es compressed-tensors "pack-quantized", no GPTQ: los nombres son
    # weight_packed / weight_scale / weight_shape, y el empaquetado va [N, K/8] con 8 nibbles por
    # int32 a lo largo de K. int4 simetrico con group_size 128 (lo dice el config).
    from safetensors import safe_open
    base = glob.glob("/root/.cache/huggingface/hub/models--noon-at-cgn--*/snapshots/*")[0]
    idx = json.load(open(os.path.join(base, "model.safetensors.index.json")))["weight_map"]
    qs = sorted(k for k in idx if k.endswith(".weight_packed") and ".layers." in k)
    paso = max(1, len(qs) // n)
    salida = []
    for nombre in qs[::paso][:n]:
        pref = nombre[: -len(".weight_packed")]
        with safe_open(os.path.join(base, idx[nombre]), framework="pt") as h:
            qw = h.get_tensor(nombre)                       # [N, K/8] int32
            sc = h.get_tensor(pref + ".weight_scale")       # [N, K/G] fp16
        N, K = qw.shape[0], qw.shape[1] * 8
        q = torch.zeros((N, K), dtype=torch.int32)
        for i in range(8):
            q[:, i::8] = (qw >> (4 * i)) & 0xF
        # El nibble es complemento a dos de 4 bits: [0,15] representa [-8,7], no valor+8.
        qs = torch.where(q >= 8, q - 16, q).float()
        w = qs * sc.float().repeat_interleave(G, dim=1)
        salida.append((pref, w, sc))
    return salida


def err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / a.norm())


def esquema_actual(w: torch.Tensor, sc: torch.Tensor | None = None):
    """Lo de hoy: q4 simetrico por grupo con la escala del checkpoint.

    Cuando se le pasa la escala REAL del checkpoint esto es el control: re-cuantizar un peso que
    ya vive en esa grilla tiene que dar error ~0. Si no da ~0, el resto de la tabla no significa
    nada — es la forma de saber que la reconstruccion del peso es correcta.
    """
    n, k = w.shape
    g = w.reshape(n, k // G, G)
    if sc is None:
        s = g.abs().amax(2).clamp_min(1e-9) / 8.0
    else:
        # OJO: la MITAD de las escalas de este checkpoint son NEGATIVAS (es AutoRound, y por eso
        # existe PN130). Un clamp_min aca las convierte en +1e-9 y destruye el peso entero: el
        # control pasa de ~0 a 5e9. Se acota el modulo, no el valor.
        s = sc.float()
        s = torch.where(s.abs() < 1e-9, torch.full_like(s, 1e-9), s)
    q = (g / s.unsqueeze(2)).round().clamp(-8, 7)
    return (q * s.unsqueeze(2)).reshape(n, k)


def esquema_liquid(w: torch.Tensor):
    """LiquidQuant: progresivo fp16 -> int8 por canal -> uint4 por grupo con escala UINT8.

    La cadena entera, tal cual la hace el kernel:
        q8  = round(w / t)                      t = escala por canal, fp16 (una por fila)
        u8  = q8 - min(q8)                      pasa a espacio unsigned
        s   = ceil(max(u8) / 15)                escala del grupo, tiene que ser UINT8
        q4  = round(u8 / s)                     en [0,15]
        rec = ((q4 * s + a) ^ 0x80) * t         con a = 128 + min(q8)
    El ^0x80 es sumar/restar 128 en complemento a dos, que es lo que hace el kernel con el XOR.
    """
    n, k = w.shape
    t = w.abs().amax(1, keepdim=True).clamp_min(1e-9) / 127.0      # por canal
    q8 = (w / t).round().clamp(-128, 127)
    g8 = q8.reshape(n, k // G, G)
    mn = g8.amin(2, keepdim=True)
    u8 = g8 - mn                                                    # [0, 255]
    s = (u8.amax(2, keepdim=True) / 15.0).ceil().clamp(1, 255)      # escala UINT8
    q4 = (u8 / s).round().clamp(0, 15)
    rec8 = (q4 * s + mn).clamp(-128, 127)                           # de vuelta a int8
    return (rec8.reshape(n, k) * t)


def esquema_liquid_s_fp(w: torch.Tensor):
    """Variante: misma cadena pero SIN forzar la escala del grupo a entero.

    Sirve para separar cuanto del error viene de la cadena progresiva (el paso por int8) y cuanto
    de obligar a que la escala sea uint8, que es lo que el IMAD necesita.
    """
    n, k = w.shape
    t = w.abs().amax(1, keepdim=True).clamp_min(1e-9) / 127.0
    q8 = (w / t).round().clamp(-128, 127)
    g8 = q8.reshape(n, k // G, G)
    mn = g8.amin(2, keepdim=True)
    u8 = g8 - mn
    s = (u8.amax(2, keepdim=True) / 15.0).clamp_min(1e-9)           # sin ceil: fp
    q4 = (u8 / s).round().clamp(0, 15)
    return ((q4 * s + mn).reshape(n, k) * t)


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pesos = cargar_pesos(N_TENSORES)
    print(f"error relativo ||w' - w|| / ||w||, sobre el peso que el modelo usa HOY\n")
    print(f"  {'tensor':<46}{'control':>10}{'liquid':>10}{'liquid s=fp':>13}")
    print(f"  {'(control = re-cuantizar con el esquema de hoy; tiene que dar ~0)':<79}")
    acum = [0.0, 0.0, 0.0]
    for nombre, w, sc in pesos:
        w = w.to(dev).float()
        sc = sc.to(dev)
        e0 = err(w, esquema_actual(w, sc))
        e1 = err(w, esquema_liquid(w))
        e2 = err(w, esquema_liquid_s_fp(w))
        del w
        torch.cuda.empty_cache()
        for i, e in enumerate((e0, e1, e2)):
            acum[i] += e
        corto = nombre.replace("model.language_model.layers.", "L")
        print(f"  {corto[:46]:<46}{e0:>10.4f}{e1:>10.4f}{e2:>13.4f}")
    n = len(pesos)
    print(f"  {'PROMEDIO':<46}{acum[0]/n:>10.4f}{acum[1]/n:>10.4f}{acum[2]/n:>13.4f}")
    print(f"\n  el control da {acum[0]/n:.4f}: la reconstruccion del peso es exacta, asi que la "
          f"comparacion vale")
    print(f"  LiquidQuant AGREGA {acum[1]/n*100:.2f}% de error relativo sobre el peso de hoy")
    print(f"  y el {(acum[1]-acum[2])/max(acum[1],1e-9)*100:.0f}% de eso es por forzar la escala "
          f"a uint8 — que es justo lo que habilita el IMAD")


if __name__ == "__main__":
    main()
