"""SK-18 fase A0 — numerica de atencion ENTERA sobre q/k/v reales (volcados PN123).

Referencia: atencion float exacta (causal) sobre q/k/v originales.
Variantes (todo lo que el kernel haria en enteros, emulado en torch):
  * rotacion Hadamard aleatoria de q y k (PN126) y opcionalmente de v
    (con v rotada la salida sale rotada y se des-rota: H^T, exacto);
  * Q, K, V cuantizadas por token-cabeza a int8 o int4 (simetrico);
  * suavizado estilo SageAttention2 para Q.K en int4: Q - media por bloque,
    K - media global, y el termino de correccion exacto se suma;
  * IndexSoftmax (IntAttention): distancia al maximo en unidades reales,
    recorte en c, tabla de L entradas uint8 de exp(-c*i/(L-1)), normalizacion
    entera; pesos uint8; salida = sum(w * v) / sum(w).
Mide el error relativo de la salida de atencion (norma L2) por consulta.
"""
import math, os, sys, torch

torch.manual_seed(0)
D, HKV, HQ = 256, 2, 12
G = HQ // HKV
DIR = "/kv"


def cargar(capa):
    qs, ks, vs = [], [], []
    for i in (1, 2, 3):
        d = torch.load(f"{DIR}/capa{capa:02d}_{i}.pt")
        qs.append(d["q"]); ks.append(d["k"]); vs.append(d["v"])
    return (torch.cat(qs).view(-1, HQ, D).float(), torch.cat(ks).view(-1, HKV, D).float(),
            torch.cat(vs).view(-1, HKV, D).float())


def hadamard(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    g = torch.Generator().manual_seed(126)
    s = torch.randint(0, 2, (n,), generator=g).float() * 2 - 1
    return H / math.sqrt(n) * s[None, :]


R = hadamard(D)


def cuant(x, bits):
    """simetrico por ultima dim (token-cabeza): -> (entero float, escala)."""
    m = (1 << (bits - 1)) - 1
    s = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / m
    return (x / s).round().clamp(-m, m), s


def atencion(qx, kx, vx, pos, softmax):
    """qx [B, HQ, D] (una por pos), kx/vx [T, HKV, D]. softmax(logits [B,HKV,G,T], mask) -> pesos."""
    T = int(pos.max()) + 1
    q = qx.view(len(pos), HKV, G, D)
    logits = torch.einsum("bkgd,tkd->bkgt", q, kx[:T]) / math.sqrt(D)
    mask = torch.arange(T)[None, :] > pos[:, None]
    w = softmax(logits, mask[:, None, None, :])
    return torch.einsum("bkgt,tkd->bkgd", w, vx[:T]).reshape(len(pos), HQ, D)


def softmax_float(logits, mask):
    return logits.masked_fill(mask, float("-inf")).softmax(-1)


def index_softmax(L, c):
    """IntAttention: distancia al maximo, recorte, tabla uint8 de L entradas, normalizacion entera."""
    tabla = torch.round(255 * torch.exp(-c * torch.arange(L).float() / (L - 1)))
    tabla[-1] = 0.0

    def f(logits, mask):
        lg = logits.masked_fill(mask, float("-inf"))
        mx = lg.amax(-1, keepdim=True)
        d = (mx - lg).clamp(max=c)
        idx = torch.round(d * (L - 1) / c).long().clamp(0, L - 1)
        w = tabla[idx].masked_fill(mask, 0.0)                      # uint8
        S = w.sum(-1, keepdim=True).clamp_min(1.0)                 # acumulador entero
        return w / S
    return f


def index_softmax_ancho(L, c, tbits, wbits=None):
    """Tabla de tbits (acumulador entero ancho) y, si wbits, pesos normalizados
    cuantizados a wbits para el GEMM pesos x V (redondeo con arrastre del error
    por fila para conservar la masa de la cola)."""
    maxv = (1 << tbits) - 1
    tabla = torch.floor(maxv * torch.exp(-c * torch.arange(L).float() / (L - 1)) + 0.5)

    def f(logits, mask):
        lg = logits.masked_fill(mask, float("-inf"))
        mx = lg.amax(-1, keepdim=True)
        d = (mx - lg).clamp(max=c)
        idx = torch.round(d * (L - 1) / c).long().clamp(0, L - 1)
        w = tabla[idx].masked_fill(mask, 0.0)
        if c < 1e9:
            w = torch.where(d >= c, torch.zeros_like(w), w)
        S = w.sum(-1, keepdim=True).clamp_min(1.0)
        wn = w / S
        if wbits:
            m = (1 << wbits) - 1
            wmax = wn.amax(-1, keepdim=True).clamp_min(1e-12)
            wq = torch.round(wn / wmax * m)
            # masa perdida por redondeo: se reparte proporcional (renormalizar)
            wn = wq / wq.sum(-1, keepdim=True).clamp_min(1.0)
        return wn
    return f


def variante(q, k, v, pos, qbits=None, kbits=None, vbits=None, rot=True, rot_v=False,
             suav=False, softmax=softmax_float):
    qx, kx, vx = q, k, v
    if rot:
        qx = qx @ R.t(); kx = kx @ R.t()
    if rot_v:
        vx = vx @ R.t()
    if suav:                       # SageAttention2: K - media global, Q - media por "bloque" (fila)
        km = kx.mean(0, keepdim=True)              # [1, HKV, D]
        kx_s = kx - km
        q4 = qx.view(-1, HKV, G, D)
        qm = q4.mean(-1, keepdim=True)             # por token-cabeza (bloque = fila)
        qx_s = (q4 - qm).view_as(qx)
    else:
        kx_s, qx_s, km = kx, qx, None
    if qbits:
        qi, qs = cuant(qx_s, qbits); qx_q = qi * qs
    else:
        qx_q = qx_s
    if kbits:
        ki, ks = cuant(kx_s, kbits); kx_q = ki * ks
    else:
        kx_q = kx_s
    if suav:
        # q.k = (q-qm).(k-km) + (q-qm).km + qm.(k)   ; los terminos de correccion se calculan exactos
        qk_extra_q = qx_s.view(-1, HKV, G, D)                       # para (q-qm).km
        qm_full = qx.view(-1, HKV, G, D) - qx_s.view(-1, HKV, G, D)
    if vbits:
        vi, vs = cuant(vx, vbits); vx_q = vi * vs
    else:
        vx_q = vx

    if not suav:
        o = atencion(qx_q, kx_q, vx_q, pos, softmax)
    else:
        T = int(pos.max()) + 1
        qq = qx_q.view(len(pos), HKV, G, D)
        logits = torch.einsum("bkgd,tkd->bkgt", qq, kx_q[:T])
        corr1 = torch.einsum("bkgd,kd->bkg", qk_extra_q, km[0])[..., None]
        corr2 = torch.einsum("bkgd,tkd->bkgt", qm_full, kx[:T])
        logits = (logits + corr1 + corr2) / math.sqrt(D)
        mask = torch.arange(T)[None, :] > pos[:, None]
        w = softmax(logits, mask[:, None, None, :])
        o = torch.einsum("bkgt,tkd->bkgd", w, vx_q[:T]).reshape(len(pos), HQ, D)
    if rot_v:
        o = o @ R            # des-rotar la salida
    return o


def error(o, ref):
    return 100 * ((o - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-6)).mean().item()


for capa in [int(c) for c in (sys.argv[1:] or ["3", "35"])]:
    q, k, v = cargar(capa)
    N = k.shape[0]
    pos = torch.linspace(8000, N - 1, 48).long()
    ref = atencion(q[pos], k, v, pos, softmax_float)
    qp = q[pos]
    print(f"\ncapa {capa}  (N={N}, 48 consultas)", flush=True)
    casos = [
        ("IndexSoftmax L=32 c=6.6 (resto float)", dict(rot=False, softmax=index_softmax(32, 6.6))),
        ("IndexSoftmax L=64 c=8", dict(rot=False, softmax=index_softmax(64, 8.0))),
        ("IndexSoftmax L=256 c=10", dict(rot=False, softmax=index_softmax(256, 10.0))),
        ("Q8 K8 Hadamard, softmax float, V float", dict(qbits=8, kbits=8)),
        ("Q8 K8 V8 Hadamard + IdxSoftmax L=64", dict(qbits=8, kbits=8, vbits=8, softmax=index_softmax(64, 8.0))),
        ("Q8 K8 V8 Had(+V) + IdxSoftmax L=64", dict(qbits=8, kbits=8, vbits=8, rot_v=True, softmax=index_softmax(64, 8.0))),
        ("Q8 K4 V4 Had(+V) + IdxSoftmax L=64", dict(qbits=8, kbits=4, vbits=4, rot_v=True, softmax=index_softmax(64, 8.0))),
        ("Q4 K4 Hadamard, softmax float", dict(qbits=4, kbits=4)),
        ("Q4 K4 Had + suavizado Sage2, softmax float", dict(qbits=4, kbits=4, suav=True)),
        ("Q4 K4 V4 Had(+V)+Sage2 + IdxSoftmax L=64", dict(qbits=4, kbits=4, vbits=4, rot_v=True, suav=True, softmax=index_softmax(64, 8.0))),
    ]
    casos += [
        ("IdxSm tabla u16 L=256 c=11", dict(rot=False, softmax=index_softmax_ancho(256, 11.0, 16))),
        ("IdxSm tabla u24 L=1024 c=16", dict(rot=False, softmax=index_softmax_ancho(1024, 16.0, 24))),
        ("IdxSm tabla u31 L=4096 c=21", dict(rot=False, softmax=index_softmax_ancho(4096, 21.0, 31))),
        ("IdxSm u31 L=4096 c=21, pesos a 8 bits", dict(rot=False, softmax=index_softmax_ancho(4096, 21.0, 31, 8))),
        ("IdxSm u31 L=4096 c=21, pesos a 12 bits", dict(rot=False, softmax=index_softmax_ancho(4096, 21.0, 31, 12))),
        ("IdxSm u31 L=4096 c=21, pesos a 16 bits", dict(rot=False, softmax=index_softmax_ancho(4096, 21.0, 31, 16))),
        ("Q8K8V8 Had(+V) + IdxSm u31 L=4096 c=21", dict(qbits=8, kbits=8, vbits=8, rot_v=True, softmax=index_softmax_ancho(4096, 21.0, 31))),
        ("Q8K4V4 Had(+V) + IdxSm u31 L=4096 c=21", dict(qbits=8, kbits=4, vbits=4, rot_v=True, softmax=index_softmax_ancho(4096, 21.0, 31))),
        ("Q4K4V4 Had(+V)+Sage2 + IdxSm u31", dict(qbits=4, kbits=4, vbits=4, rot_v=True, suav=True, softmax=index_softmax_ancho(4096, 21.0, 31))),
    ]
    for nombre, kw in casos:
        o = variante(qp, k, v, pos, **kw)
        print(f"  {nombre:46s} error salida {error(o, ref):7.3f}%", flush=True)
