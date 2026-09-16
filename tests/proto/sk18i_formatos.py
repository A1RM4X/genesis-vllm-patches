"""int4 KV: formatos ENTEROS que entran en los 8 bytes de escala por token-cabeza.

Simula exactamente lo que guardaria el kernel (escalas cuantizadas incluidas) y mide el error
relativo de la salida de atencion con softmax float y q en fp16 (equivalente al camino de
prefill de PN131, que es el piso de error del decode).

Presupuesto: 8 B por token-cabeza. Formatos probados:
  F0  (el que corre)  kmax int16 | 4 ratios uint8 (g64 simetrica) | svf int16
  F1  asimetrica      kmax 12b + vzp 4b | 4 x (ratio 4b | zp 4b) | svf int16 + V asimetrica
  F2  zp por grupo    kmax int16 (escala unica) | 4 zp uint8 | svf int16, V asimetrica (zp en kmax)
  F3  solo V asim     kmax 12b + vzp 4b | 4 ratios uint8 | svf int16
  F4  K asim, V sim   kmax 12b | 4 x (ratio 4b | zp 4b) | svf int16
"""
import math, sys, torch
import sk18_a0bis_ptx as A
dev = "cuda"; D = 256; G = 6; HKV = 2
torch.manual_seed(0)
H = A.R.to(dev).float()


def rot(x):
    return x @ H.T


def redondear(x):
    return torch.round(x)


def k_sim_g64(K, bits_ratio=8):
    """simetrica por grupo de 64 dims, escala = kmax * ratio / max_ratio (el formato actual)."""
    n = (1 << bits_ratio) - 1
    Kg = K.reshape(-1, 4, 64)
    mg = Kg.abs().amax(-1, keepdim=True)                       # [N,4,1]
    mt = mg.amax(1, keepdim=True)
    r = torch.ceil(mg / mt.clamp_min(1e-9) * n).clamp(1, n)
    mef = mt * r / n
    s = (mef / 7).clamp_min(1e-9)
    q = redondear(Kg / s).clamp(-7, 7)
    return (q * s).reshape(K.shape)


def k_asim_g64(K, bits_ratio=4, kmax_bits=12):
    """asimetrica por grupo: nib en [-8,7], x ~ (nib - zp) * s_g, s_g = smax * ratio / 2^b."""
    n = (1 << bits_ratio)
    Kg = K.reshape(-1, 4, 64)
    lo = Kg.amin(-1, keepdim=True); hi = Kg.amax(-1, keepdim=True)
    s = ((hi - lo) / 15).clamp_min(1e-9)
    smax = s.amax(1, keepdim=True)
    # smax se guarda con kmax_bits bits (relativo a la referencia de la capa: aca solo su precision)
    esc = 2 ** torch.ceil(torch.log2(smax))
    smax = (redondear(smax / esc * (2 ** kmax_bits - 1)) / (2 ** kmax_bits - 1)) * esc
    r = torch.ceil(s / smax.clamp_min(1e-9) * n).clamp(1, n)
    s = smax * r / n
    zp = redondear(-lo / s - 8).clamp(-8, 7)                   # cero en el dominio del nibble
    q = redondear(Kg / s + zp).clamp(-8, 7)
    return ((q - zp) * s).reshape(K.shape)


def k_asim_zp8(K, kmax_bits=16):
    """escala unica por token-cabeza, cero por grupo con 8 bits."""
    Kg = K.reshape(-1, 4, 64)
    lo = Kg.amin(-1, keepdim=True); hi = Kg.amax(-1, keepdim=True)
    s = ((hi - lo) / 15).amax(1, keepdim=True).clamp_min(1e-9)
    esc = 2 ** torch.ceil(torch.log2(s))
    s = (redondear(s / esc * (2 ** kmax_bits - 1)) / (2 ** kmax_bits - 1)) * esc
    zp = redondear(-lo / s - 8).clamp(-128, 127)
    q = redondear(Kg / s + zp).clamp(-8, 7)
    return ((q - zp) * s).reshape(K.shape)


def v_sim(V, bits=16):
    m = V.abs().amax(-1, keepdim=True).clamp_min(1e-9)
    s = (m / 7)
    return redondear(V / s).clamp(-7, 7) * s


def v_asim(V, zp_bits=4):
    lo = V.amin(-1, keepdim=True); hi = V.amax(-1, keepdim=True)
    s = ((hi - lo) / 15).clamp_min(1e-9)
    zp = redondear(-lo / s - 8).clamp(-8, 7)
    q = redondear(V / s + zp).clamp(-8, 7)
    return (q - zp) * s


FORMATOS = {
    "F0 K sim g64 r8 | V sim (lo que corre)": (lambda K: k_sim_g64(K, 8), v_sim),
    "F1 K asim g64 r4/zp4 | V asim":          (lambda K: k_asim_g64(K, 4, 12), v_asim),
    "F2 K asim escala unica zp8 | V asim":    (lambda K: k_asim_zp8(K), v_asim),
    "F3 K sim g64 r8 | V asim":               (lambda K: k_sim_g64(K, 8), v_asim),
    "F4 K asim g64 r4/zp4 | V sim":           (lambda K: k_asim_g64(K, 4, 12), v_sim),
    "F5 K asim g64 r8/zp4 (no entra) | V asim": (lambda K: k_asim_g64(K, 8, 16), v_asim),
}


def atn(q, k, v, lim):
    lg = (q @ k.transpose(-1, -2)) / math.sqrt(D)
    mask = torch.arange(k.shape[-2], device=dev)[None] > lim[:, None]
    return lg.masked_fill(mask, float("-inf")).softmax(-1) @ v


for capa in [int(c) for c in (sys.argv[1:] or ["3", "35"])]:
    q, k, v = A.cargar(capa); Nt = k.shape[0]
    posl = [torch.arange(p, p + 4, device=dev) for p in torch.linspace(8000, Nt - 5, 8).long().tolist()]
    res = {}
    for h in range(HKV):
        K = rot(k[:, h].float()); V = rot(v[:, h].float())
        for nombre, (fk, fv) in FORMATOS.items():
            Kq = fk(K); Vq = fv(V)
            e = []
            for pos in posl:
                N = int(pos.max()) + 1
                Q = rot(q[pos][:, h * G:(h + 1) * G].reshape(-1, D).float())
                lim = pos.repeat_interleave(G)
                r = A.referencia(q, k, v, pos).float()[:, h * G:(h + 1) * G].reshape(-1, D)
                o = atn(Q, Kq[:N], Vq[:N], lim) @ H
                e.append(((o - r).norm(dim=-1) / r.norm(dim=-1).clamp_min(1e-6)).mean().item())
            res.setdefault(nombre, []).append(sum(e) / len(e))
    print(f"\ncapa {capa}")
    for kx, vx in res.items():
        print(f"  {kx:45s} {100 * sum(vx) / len(vx):6.3f}%", flush=True)
