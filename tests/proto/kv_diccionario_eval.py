"""Evaluacion de compresion de la KV de atencion con q/k/v REALES (volcados PN123).

Metrica principal: error relativo de la SALIDA de la atencion
    out_t = softmax(q_t . K_{<=t} / sqrt(d)) V_{<=t}
para queries reales muestreadas en el ultimo tramo (contexto causal completo),
mas la coincidencia del top-8 de los pesos de atencion. El error de los vectores
por si solo engana: lo que importa es lo que la atencion lee.

Codebooks entrenados SOLO con el primer tramo (tokens 0..7487) y evaluados sobre
todo el contexto: el tramo 3 es fuera de la muestra.
"""
import math
import sys
import time

import torch

torch.manual_seed(0)
dev = "cuda"
D, HKV, HQ = 256, 2, 12
DIR = "/kv-offload/pn123_qkv"


def cargar(capa):
    qs, ks, vs = [], [], []
    for i in (1, 2, 3):
        d = torch.load(f"{DIR}/capa{capa:02d}_{i}.pt")
        qs.append(d["q"]); ks.append(d["k"]); vs.append(d["v"])
    q = torch.cat(qs).view(-1, HQ, D)
    k = torch.cat(ks).view(-1, HKV, D)
    v = torch.cat(vs).view(-1, HKV, D)
    return q, k.to(dev).float(), v.to(dev).float()


def hadamard(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(n)).to(dev)


H = hadamard(D)
SIGNOS = (torch.randint(0, 2, (D,), device=dev) * 2 - 1).float()


def rot(x):      # RHT: H * diag(signos) * x
    return (x * SIGNOS) @ H.T


def desrot(y):
    return (y @ H) * SIGNOS


# ───────────────────────────── escalares ─────────────────────────────
def fp8_tensor(x):
    return x.to(torch.float8_e4m3fn).float()      # escala 1.0, como hoy


def int_por_token(x, bits, rotar):
    y = rot(x) if rotar else x
    lo = y.amin(-1, keepdim=True); hi = y.amax(-1, keepdim=True)
    niveles = 2 ** bits - 1
    s = (hi - lo).clamp_min(1e-6) / niveles
    yq = ((y - lo) / s).round().clamp(0, niveles) * s + lo
    return desrot(yq) if rotar else yq


# ───────────────────────────── codebooks ─────────────────────────────
def kmeans(x, k, iters=25):
    c = x[torch.randperm(x.shape[0], device=dev)[:k]].clone()
    for _ in range(iters):
        idx = torch.cdist(x, c).argmin(1)
        suma = torch.zeros_like(c).index_add_(0, idx, x)
        n = torch.bincount(idx, minlength=k).clamp_min(1).unsqueeze(1)
        vacios = torch.bincount(idx, minlength=k) == 0
        c = suma / n
        if vacios.any():
            c[vacios] = x[torch.randint(0, x.shape[0], (int(vacios.sum()),), device=dev)]
    return c


class PQ:
    """Product quantization tras RHT y normalizacion por token-cabeza (norma fp16).
    ds dims por subespacio, 2**bits_sub centroides por subespacio."""

    def __init__(self, entren, ds, bits_sub, rotar=True, centrar=False):
        self.rotar, self.centrar = rotar, centrar
        self.media = entren.mean(0, keepdim=True) if centrar else 0.0
        entren = entren - self.media
        y = rot(entren) if rotar else entren
        self.ds, self.k = ds, 2 ** bits_sub
        yn = y / y.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        m = D // ds
        sub = yn.view(-1, m, ds)
        self.cb = torch.stack([kmeans(sub[:, j], self.k) for j in range(m)])  # [m,k,ds]
        self.bits = m * bits_sub + 16

    def __call__(self, x):
        x = x - self.media
        y = rot(x) if self.rotar else x
        nrm = y.norm(dim=-1, keepdim=True)
        yn = (y / nrm.clamp_min(1e-6)).view(-1, D // self.ds, self.ds)
        out = torch.empty_like(yn)
        for j in range(yn.shape[1]):
            idx = torch.cdist(yn[:, j], self.cb[j]).argmin(1)
            out[:, j] = self.cb[j][idx]
        o = out.view(-1, D) * nrm
        return (desrot(o) if self.rotar else o) + self.media


class RVQ:
    """Dos etapas de PQ sobre el residuo (tras RHT + norma)."""

    def __init__(self, entren, ds, bits_sub):
        self.a = PQ(entren, ds, bits_sub)
        res = entren - self.a(entren)
        self.b = PQ(res, ds, bits_sub)
        self.bits = self.a.bits + self.b.bits

    def __call__(self, x):
        p = self.a(x)
        return p + self.b(x - p)


class Disperso:
    """Diccionario universal tipo Lexico: N atomos unitarios, s coeficientes por
    vector (OMP en lote). Diccionario aprendido con MOD sobre el tramo 1.
    Bits: s * (log2 N indice + 8 coeficiente fp8)."""

    def __init__(self, entren, n_atomos, s, iters=8):
        self.s = s
        Dic = entren[torch.randperm(entren.shape[0], device=dev)[:n_atomos]].clone()
        Dic = Dic / Dic.norm(dim=1, keepdim=True).clamp_min(1e-6)
        x = entren[torch.randperm(entren.shape[0], device=dev)[:(6000 if s <= 24 else 2500)]]
        for _ in range(iters):
            A = self._codigos(x, Dic)                    # [n, N] denso con s no-ceros
            # MOD con ridge: x ≈ A Dic. Sin regularizar, atomos sin uso dejan
            # la normal singular y aparecen NaN.
            G = A.T @ A + 1e-3 * torch.eye(A.shape[1], device=dev)
            Dic = torch.linalg.solve(G, A.T @ x)
            Dic = Dic / Dic.norm(dim=1, keepdim=True).clamp_min(1e-6)
        self.Dic = Dic
        self.bits = s * (math.ceil(math.log2(n_atomos)) + 8)

    def _codigos(self, x, Dic):
        n = x.shape[0]
        res = x.clone()
        sel = torch.zeros(n, 0, dtype=torch.long, device=dev)
        for _ in range(self.s):
            corr = res @ Dic.T
            if sel.shape[1]:
                corr.scatter_(1, sel, 0.0)
            nuevo = corr.abs().argmax(1, keepdim=True)
            sel = torch.cat([sel, nuevo], 1)
            At = Dic[sel]                                  # [n, j, D]
            G = At @ At.transpose(1, 2) + 1e-4 * torch.eye(At.shape[1], device=dev)
            coef = torch.linalg.solve(G, At @ x.unsqueeze(-1)).squeeze(-1)
            res = x - (coef.unsqueeze(-1) * At).sum(1)
        A = torch.zeros(n, Dic.shape[0], device=dev)
        A.scatter_(1, sel, coef)
        return A

    def __call__(self, x):
        salida = []
        paso = 2048 if self.s <= 24 else 768
        for lo in range(0, x.shape[0], paso):
            xb = x[lo:lo + paso]
            A = self._codigos(xb, self.Dic)
            # coeficientes a fp8 por vector (escala = max abs / 127 en 8 bits)
            s = A.abs().amax(1, keepdim=True).clamp_min(1e-6) / 127
            A = (A / s).round() * s
            salida.append(A @ self.Dic)
        return torch.cat(salida)


# ───────────────────────────── metrica ─────────────────────────────
def salida_atencion(q, k, v, pos):
    # GQA sin duplicar keys: q [b, HKV, HQ/HKV, D] contra k [T, HKV, D].
    outs, tops = [], []
    g = HQ // HKV
    for lo in range(0, len(pos), 4):
        p = pos[lo:lo + 4]
        T = int(p.max()) + 1
        qq = q[p].to(dev).float().view(len(p), HKV, g, D)
        sc = torch.einsum("bkgd,tkd->bkgt", qq, k[:T]) / math.sqrt(D)
        mask = torch.arange(T, device=dev)[None, :] > p.to(dev)[:, None]
        sc = sc.masked_fill(mask[:, None, None, :], float("-inf"))
        w = sc.softmax(-1)
        del sc
        outs.append(torch.einsum("bkgt,tkd->bkgd", w, v[:T]).reshape(len(p), HQ, D))
        tops.append(w.topk(8, -1).indices.reshape(len(p), HQ, 8))
        del w
    return torch.cat(outs), torch.cat(tops)


def evaluar(capa):
    q, k, v = cargar(capa)
    N = k.shape[0]
    pos = torch.linspace(16000, N - 1, 192).long()
    ref, top_ref = salida_atencion(q, k, v, pos)
    ek = k[:7488].reshape(-1, D); ev = v[:7488].reshape(-1, D)

    def comprimir(fk, fv):
        kq = fk(k.reshape(-1, D)).view_as(k)
        vq = fv(v.reshape(-1, D)).view_as(v)
        return kq, vq

    metodos = [
        ("int4 + Hadamard (vLLM)", 4 + 32 / D, lambda: (lambda x: int_por_token(x, 4, True),) * 2),
        ("int2 + Hadamard", 2 + 32 / D, lambda: (lambda x: int_por_token(x, 2, True),) * 2),
    ]
    for ds, b, et in ((4, 8, "2 bits"), (2, 8, "4 bits")):
        for rotar, centrar in ((True, False), (False, False), (False, True)):
            pk = PQ(ek, ds, b, rotar, centrar); pv = PQ(ev, ds, b, rotar, centrar)
            nom = f"PQ {ds}d/256 {et} {'rot' if rotar else 'sin rot'}{' +media' if centrar else ''}"
            metodos.append((nom, pk.bits / D, (lambda a, c: (lambda: (a, c)))(pk, pv)))
    print(f"\n=== capa {capa}: {N} tokens, 192 queries x {HQ} cabezas en el tramo 16000..{N-1} ===")
    print(f"{'metodo':36} {'bits/elem':>9} {'x vs fp16':>9} {'err salida':>11} {'top8 igual':>10} {'err K':>8} {'err V':>8}")
    for nombre, bits, mk in metodos:
        t0 = time.time()
        fk, fv = mk()
        kq, vq = comprimir(fk, fv)
        out, top = salida_atencion(q, kq, vq, pos)
        err = ((out - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-6)).mean().item()
        igual = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in
                    zip(top.reshape(-1, 8), top_ref.reshape(-1, 8))) / top.reshape(-1, 8).numel()
        ek_ = ((kq - k).norm() / k.norm()).item(); ev_ = ((vq - v).norm() / v.norm()).item()
        print(f"{nombre:36} {bits:>9.2f} {16 / bits:>8.1f}x {100 * err:>10.2f}% {100 * igual:>9.1f}% "
              f"{100 * ek_:>7.2f}% {100 * ev_:>7.2f}%  ({time.time() - t0:.0f}s)")
        del kq, vq
        torch.cuda.empty_cache()


if __name__ == "__main__":
    for c in (int(x) for x in (sys.argv[1:] or ["3", "35"])):
        evaluar(c)
