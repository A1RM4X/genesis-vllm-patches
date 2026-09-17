"""Banco offline de metodos ESTATICOS de cuantizacion de activacion.

La restriccion que manda: todo tiene que resolverse en boot y quedar cacheado. Nada de pasadas
extra ni de kernels mas gordos en el camino caliente. Eso descarta de entrada cualquier cosa que
necesite una reduccion sobre la fila en tiempo de ejecucion.

Lo que SI se puede hacer en boot, gratis, con este checkpoint (GPTQ int4, ``group_size=128``,
``desc_act=False``):

  * una escala por capa (un float);
  * un factor de suavizado por GRUPO de 128 canales de entrada. Y es exacto: el peso queda
    ``w'[j][out] = w[j][out] * s_g``, y como dentro del grupo ``s`` es constante, alcanza con
    multiplicar la escala de grupo que el checkpoint YA guarda. Del lado de la activacion,
    ``x_j / s_g`` se pliega en el peso de la RMSNorm previa, que es por canal. Cero re-cuantizacion
    y cero costo en ejecucion.
  * una rotacion, si se pliega en los pesos — pero eso SI exige re-cuantizar, asi que aca se mide
    solo como cota superior de referencia.

Metrica: error relativo de la activacion cuantizada, ``||q(x) - x|| / ||x||``. Es el que manda
para el GEMM: el error de ``x @ W`` es proporcional en esperanza para cualquier W.

Uso: estaticos_offline.py <dir_muestras> [n_capas_por_tipo]
"""
import glob
import math
import os
import statistics
import sys

import torch

DIR = sys.argv[1] if len(sys.argv) > 1 else "muestras"
POR_TIPO = int(sys.argv[2]) if len(sys.argv) > 2 else 4
G = 128            # group_size del checkpoint


def _limpiar(x: torch.Tensor) -> torch.Tensor:
    """Saca las filas de relleno del prefill chunked: vienen con memoria sin inicializar (NaN) o
    en cero, y envenenan cualquier estadistica."""
    bien = torch.isfinite(x).all(1) & (x.abs().amax(1) > 0)
    return x[bien]


def err(x: torch.Tensor, xq: torch.Tensor) -> float:
    return float((xq - x).norm() / x.norm())


# ── metodos ──────────────────────────────────────────────────────────────────────────────────
def dinamico_token(x):
    """Lo de hoy: una escala por fila, calculada en caliente. NO es estatico; es la referencia."""
    s = x.abs().amax(-1, keepdim=True).clamp_min(1e-9) / 127.0
    return (x / s).round().clamp_(-127, 127) * s


def estatico_tensor(x, cal):
    """Una escala por capa."""
    s = cal["amax"] / 127.0
    return (x / s).round().clamp_(-127, 127) * s


def estatico_percentil(x, cal, p=99.9):
    """Una escala por capa, pero cortando la cola: recorta el p% mas alto a cambio de resolucion."""
    s = cal[f"p{p}"] / 127.0
    return (x / s).round().clamp_(-127, 127) * s


def suave_grupo_robusto(x, cal, alfa=1.0, p="p999"):
    """Igual, pero el factor del grupo sale de un percentil alto en vez del maximo: asi un solo
    token raro no infla el factor de todo el grupo."""
    s_g = cal[f"grupo_{p}"].clamp_min(1e-9) ** alfa
    s_g = s_g / s_g.mean()
    rep = s_g.repeat_interleave(G)[: x.shape[1]]
    xs = x / rep
    s = xs.abs().amax() / 127.0
    return ((xs / s).round().clamp_(-127, 127) * s) * rep


def suave_grupo_recorte(x, cal, alfa=1.0, q=0.9999):
    """Suavizado por grupo + recorte de la cola: la escala final sale de un percentil del tensor ya
    suavizado, no de su maximo."""
    s_g = cal["grupo_p100"].clamp_min(1e-9) ** alfa
    s_g = s_g / s_g.mean()
    rep = s_g.repeat_interleave(G)[: x.shape[1]]
    xs = x / rep
    pl = xs.abs().flatten()
    idx = torch.randperm(pl.numel(), device=pl.device)[:1000000]
    s = float(torch.quantile(pl[idx].float(), q)) / 127.0
    return ((xs / s).round().clamp_(-127, 127) * s) * rep


def suave_grupo(x, cal, alfa=1.0, p=100.0):
    """SmoothQuant a granularidad de GRUPO: divide cada grupo de 128 canales por su propio factor
    y despues usa UNA escala por capa. El factor se pliega en las escalas de grupo del GPTQ."""
    s_g = cal["grupo_p100"] if p >= 100 else cal[f"grupo_p{p}"]
    s_g = s_g.clamp_min(1e-9) ** alfa
    s_g = s_g / s_g.mean()                       # normalizado: no cambia el rango global
    xs = x / s_g.repeat_interleave(G)[: x.shape[1]]
    s = xs.abs().amax() / 127.0
    return ((xs / s).round().clamp_(-127, 127) * s) * s_g.repeat_interleave(G)[: x.shape[1]]


def suave_canal(x, cal, alfa=1.0):
    """Lo mismo pero por CANAL: cota superior de lo que daria el suavizado (no implementable sin
    re-cuantizar, porque dentro de un grupo el factor tendria que variar)."""
    s_c = cal["canal_p100"].clamp_min(1e-9) ** alfa
    s_c = s_c / s_c.mean()
    xs = x / s_c
    s = xs.abs().amax() / 127.0
    return ((xs / s).round().clamp_(-127, 127) * s) * s_c


def hadamard(n):
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(h.shape[0])


_H = {}


def rotado(x, cal, b=128):
    """Rotacion Hadamard por bloques de b canales + escala por capa. Cota superior: plegarla en los
    pesos exige re-cuantizarlos."""
    if b not in _H:
        _H[b] = hadamard(b).to(x.dtype)
    H = _H[b].to(x.device)
    n = (x.shape[1] // b) * b
    xr = (x[:, :n].reshape(x.shape[0], -1, b) @ H).reshape(x.shape[0], n)
    s = cal[f"rot{b}_amax"] / 127.0
    q = (xr / s).round().clamp_(-127, 127) * s
    fuera = x.clone()
    fuera[:, :n] = (q.reshape(x.shape[0], -1, b) @ H.t()).reshape(x.shape[0], n)
    return fuera


def calibrar(x, b_rot=(128,)):
    """Todo lo que un metodo estatico puede saber de antemano."""
    cal = {"amax": float(x.abs().max())}
    plano = x.abs().flatten().float()
    for p in (99.9, 99.99):
        cal[f"p{p}"] = float(torch.quantile(plano[torch.randperm(plano.numel())[:1000000]], p / 100))
    n = (x.shape[1] // G) * G
    cal["grupo_p100"] = x[:, :n].abs().reshape(x.shape[0], -1, G).amax(dim=(0, 2))
    # percentil alto por grupo: robusto a un token aislado
    g = x[:, :n].abs().reshape(x.shape[0], -1, G).permute(1, 0, 2).reshape(n // G, -1).float()
    cal["grupo_p999"] = torch.quantile(g, 0.999, dim=1)
    cal["grupo_p99"] = torch.quantile(g, 0.99, dim=1)
    cal["canal_p100"] = x.abs().amax(0).clamp_min(1e-9)
    for b in b_rot:
        if b not in _H:
            _H[b] = hadamard(b).to(x.dtype)
        H = _H[b].to(x.device)
        nb = (x.shape[1] // b) * b
        cal[f"rot{b}_amax"] = float((x[:, :nb].reshape(x.shape[0], -1, b) @ H).abs().max())
    return cal


METODOS = [
    ("dinamico por token (hoy)", lambda x, c: dinamico_token(x)),
    ("estatico por capa", estatico_tensor),
    ("suave grupo a=0,75", lambda x, c: suave_grupo(x, c, 0.75)),
    ("suave grupo a=1,0", lambda x, c: suave_grupo(x, c, 1.0)),
    ("suave grupo a=1,25", lambda x, c: suave_grupo(x, c, 1.25)),
    ("suave grupo p99,9 a=1,0", lambda x, c: suave_grupo_robusto(x, c, 1.0, "p999")),
    ("suave grupo p99 a=1,0", lambda x, c: suave_grupo_robusto(x, c, 1.0, "p99")),
    ("suave grupo + recorte 0,9999", lambda x, c: suave_grupo_recorte(x, c, 1.0, 0.9999)),
    ("suave grupo + recorte 0,999", lambda x, c: suave_grupo_recorte(x, c, 1.0, 0.999)),
    ("suave canal a=1,0 (cota)", lambda x, c: suave_canal(x, c, 1.0)),
]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    archivos = sorted(glob.glob(os.path.join(DIR, "*.pt")))
    por_tipo = {}
    for f in archivos:
        t = os.path.basename(f).replace(".pt", "").split("_")[-2:]
        t = "_".join(t)
        por_tipo.setdefault(t, []).append(f)
    # capas repartidas (no todas del principio)
    elegidos = {}
    for t, fs in por_tipo.items():
        paso = max(1, len(fs) // POR_TIPO)
        elegidos[t] = fs[::paso][:POR_TIPO]

    print(f"error relativo ||q(x)-x||/||x||, mediana sobre {POR_TIPO} capas de cada tipo\n")
    enc = f"{'metodo':>34}"
    tipos = sorted(elegidos)
    for t in tipos:
        enc += f" {t[:13]:>13}"
    print(enc)

    res = {n: {} for n, _ in METODOS}
    for t in tipos:
        cals, xs = [], []
        for f in elegidos[t]:
            x = torch.load(f, map_location=dev).float()
            x = _limpiar(x)
            if x.shape[0] < 32:
                continue
            xs.append(x)
            cals.append(calibrar(x))
        if not xs:
            for nombre, _ in METODOS:
                res[nombre][t] = float("nan")
            continue
        for nombre, fn in METODOS:
            vals = []
            for x, c in zip(xs, cals):
                try:
                    vals.append(err(x, fn(x, c)))
                except Exception:
                    pass
            res[nombre][t] = statistics.median(vals) if vals else float("nan")
        del xs
        torch.cuda.empty_cache()

    for nombre, _ in METODOS:
        fila = f"{nombre:>34}"
        for t in tipos:
            fila += f" {res[nombre][t]:13.4f}"
        print(fila)


if __name__ == "__main__":
    main()
