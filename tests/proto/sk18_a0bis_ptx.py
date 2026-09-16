"""SK-18 A0-bis — decode con SALTO DE BLOQUES, calculado con los kernels PTX.

Por capa (volcados PN123), por cabeza KV:
  1. q, k rotadas con Hadamard 256 (SK-16, mma f16) y cuantizadas int8 por
     token-cabeza (SK-17-Q8b); v int8 por token-cabeza (Q8b).
  2. Scores exactos z = q8.k8 (SK-18a, mma s8) de las 48 consultas x 6 cabezas Q.
  3. Cota Quest por bloque: max_{j in bloque} q.k_j <= q+ . kmax_b + q- . kmin_b
     con kmax/kmin por dimension del bloque (SK-18a, dos GEMM s8).
  4. Pseudo-maximo (FFD): max de los scores sobre los 16 primeros tokens (sink)
     y el bloque mas reciente.
  5. Seleccion: bloques con cota (o max exacto) >= max - delta; o top-p sobre
     la masa exacta por bloque (oraculo).
  6. Softmax entero (tabla u24, c=16) + sum(w*v) sobre los bloques elegidos
     (SK-18b): las keys descartadas van con relleno -> peso 0.
Error de la salida contra la atencion float densa (q/k/v originales).
Pegamento en torch con enteros (mascaras, sumas de normalizacion); la
referencia float solo para comparar.
"""
import math, sys, time, torch
from vllm._genesis.kernels.ptx_lab import Kernel
from vllm._genesis.kernels import sk16_gemm_f16 as S16, sk17_act_int8 as S17

dev = "cuda"
D, HKV, HQ = 256, 2, 12
G = HQ // HKV
DIR = "/kv"
PAD = -(1 << 28)

k18a = Kernel("sk18a_qk_i32.cu", "sk18a_qk_i32", warps=8)
SH_A = 2 * (256 * 128 + 64 * 2 * 128)
BM_B, BN_B, BK_B, W_B = 64, 64, 128, 4
SH_B = 2 * (2 * BM_B * BK_B + BN_B * 2 * BK_B)
k18b = Kernel("sk18b_wv.cu", "sk18b_wv", warps=W_B)
C_RECORTE = 16.0
# peso de 15 bits relativo al maximo (32639 para que hi = (w+128)>>8 entre en s8)
LUT = torch.round(32639 * torch.exp(-C_RECORTE * torch.arange(1024).float() / 1023)).to(torch.int32).to(dev)
LUT[-1] = 0


def qk_ptx(A, B):
    """int8 [M,256] x int8 [N,256] -> int32 [M,N] con SK-18a."""
    A = A.contiguous(); B = B.contiguous()
    M, N = A.shape[0], B.shape[0]
    out = torch.empty(M, N, dtype=torch.int32, device=dev)
    k18a.lanzar(((N + 63) // 64, (M + 255) // 256), [A, B, out, M, N, 256], shared=SH_A)
    return out


def rotar_ptx(x):
    """float [T, h, 256] -> rotada con SK-16 (H aleatoria fija, la de PN126)."""
    T, h, _ = x.shape
    y = S16.gemm(x.reshape(-1, D).half().contiguous(), R16)
    return y.float().view(T, h, D)


def q8_ptx(x):
    """float [T, h, 256] -> (int8, escala) por token-cabeza con SK-17-Q8b."""
    T, h, _ = x.shape
    xq, sx = S17.q8b(x.reshape(-1, D).half().contiguous())
    return xq.view(T, h, D), sx.view(T, h)


def cargar(capa):
    qs, ks, vs = [], [], []
    for i in (1, 2, 3):
        d = torch.load(f"{DIR}/capa{capa:02d}_{i}.pt")
        qs.append(d["q"]); ks.append(d["k"]); vs.append(d["v"])
    return (torch.cat(qs).view(-1, HQ, D).float().to(dev), torch.cat(ks).view(-1, HKV, D).float().to(dev),
            torch.cat(vs).view(-1, HKV, D).float().to(dev))


def hadamard_aleatoria(n):
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    g = torch.Generator().manual_seed(126)
    s = torch.randint(0, 2, (n,), generator=g).double() * 2 - 1
    return (H / math.sqrt(n) * s[None, :])


R = hadamard_aleatoria(D)
R16 = R.half().to(dev)


def referencia(q, k, v, pos):
    T = int(pos.max()) + 1
    qq = q[pos].view(len(pos), HKV, G, D)
    lg = torch.einsum("bkgd,tkd->bkgt", qq, k[:T]) / math.sqrt(D)
    mask = torch.arange(T, device=dev)[None, :] > pos[:, None]
    w = lg.masked_fill(mask[:, None, None, :], float("-inf")).softmax(-1)
    return torch.einsum("bkgt,tkd->bkgd", w, v[:T]).reshape(len(pos), HQ, D)


def correr_capa(capa, bloques, deltas, tops):
    q, k, v = cargar(capa)
    N = k.shape[0]
    pos = torch.linspace(8000, N - 1, 48, device=dev).long()
    ref = referencia(q, k, v, pos)
    B = len(pos)
    # 1. rotacion + cuantizacion (PTX)
    qr = rotar_ptx(q[pos]); kr = rotar_ptx(k)
    q8, sq = q8_ptx(qr); k8, sk = q8_ptx(kr)
    v8, sv = q8_ptx(v)
    resultados = {}
    for h in range(HKV):
        Qh = q8[:, h * G:(h + 1) * G].reshape(B * G, D)          # [M, 256]
        sqh = sq[:, h * G:(h + 1) * G].reshape(B * G)
        posh = pos.repeat_interleave(G)
        M = B * G
        # 2. scores exactos (PTX)
        z = qk_ptx(Qh, k8[:, h])                                    # [M, N] int32
        skh = sk[:, h]
        skmax = skh.max()
        skf = torch.round(skh / skmax * 32767).to(torch.int64)
        zp = ((z.to(torch.int64) >> 8) * skf[None, :]) >> 11        # z entero fino (|z| <= ~12k)
        cols = torch.arange(N, device=dev)
        causal = cols[None, :] > posh[:, None]
        zp = zp.masked_fill(causal, PAD)
        # escala real de z': logit = z' * alpha_q
        alpha = 256.0 * skmax * sqh / math.sqrt(D) / 16.0           # logit = z * alpha (>>11 en vez de >>15)
        mq = torch.round(alpha * 1023 / C_RECORTE * 65536).clamp(max=(1 << 31) - 1).to(torch.int32)
        # V para SK-18b
        svh = sv[:, h]; svmax = svh.max()
        svf = torch.round(svh / svmax * 32767).to(torch.int16)
        CH = 1024
        K = ((N + CH - 1) // CH) * CH
        Vt = torch.zeros(D, K, dtype=torch.int8, device=dev); Vt[:, :N] = v8[:, h].t()
        svf_p = torch.zeros(K, dtype=torch.int16, device=dev); svf_p[:N] = svf
        # 3. cotas Quest por bloque (PTX): q+ . kmax + q- . kmin
        qpos = Qh.clamp(min=0); qneg = Qh.clamp(max=0)
        for bs in bloques:
            nb = (N + bs - 1) // bs
            kb = torch.full((nb * bs, D), 0, dtype=torch.int8, device=dev)
            kb[:N] = k8[:, h]
            kbv = kb.view(nb, bs, D)
            # relleno neutro para max/min: repetir la ultima key valida
            if nb * bs > N:
                kbv.view(-1, D)[N:] = k8[-1, h]
            kmax = kbv.amax(1).contiguous(); kmin = kbv.amin(1).contiguous()
            cota = qk_ptx(qpos.to(torch.int8), kmax).to(torch.int64) + qk_ptx(qneg.to(torch.int8), kmin).to(torch.int64)
            # la escala por key varia dentro del bloque: cota conservadora con skf maximo del bloque
            skf_b = torch.nn.functional.pad(skf, (0, nb * bs - N)).view(nb, bs).amax(1)
            cota_p = ((cota >> 8) * skf_b[None, :]) >> 11
            # max exacto por bloque (oraculo) y pseudo-maximo FFD
            zp_pad = torch.nn.functional.pad(zp, (0, nb * bs - N), value=PAD).view(M, nb, bs)
            zmax_b = zp_pad.amax(-1)
            zmax = zmax_b.amax(1)
            sink = zp[:, :16].amax(1)
            ult = ((posh // bs)).clamp(max=nb - 1)
            reciente = zmax_b.gather(1, ult[:, None]).squeeze(1)
            pseudo = torch.maximum(sink, reciente)
            bloque_actual = torch.arange(nb, device=dev)[None, :] == ult[:, None]
            bloque_sink = torch.arange(nb, device=dev)[None, :] == 0
            valido = (torch.arange(nb, device=dev)[None, :] * bs) <= posh[:, None]

            def evaluar(nombre, sel):
                sel = (sel | bloque_actual | bloque_sink) & valido
                keep = sel.repeat_interleave(bs, dim=1)[:, :N]
                Sz = zp.masked_fill(~keep, PAD)
                zm = Sz.amax(1)
                d = (zm[:, None] - Sz).clamp(min=0)
                idx = ((d * mq[:, None].to(torch.int64) + 32768) >> 16).clamp(max=1023)
                invS = torch.zeros(M, dtype=torch.int32, device=dev)
                Szp = torch.full((M, K), PAD, dtype=torch.int32, device=dev)
                Szp[:, :N] = Sz.clamp(min=PAD).to(torch.int32)
                out = torch.zeros(K // CH, M, D, dtype=torch.int32, device=dev)
                out_lo = torch.zeros(K // CH, M, D, dtype=torch.int32, device=dev)
                grid = ((D + BN_B - 1) // BN_B, ((M + BM_B - 1) // BM_B) * (K // CH))
                k18b.lanzar(grid, [Szp.contiguous(), zm.to(torch.int32).contiguous(), mq.contiguous(), invS,
                                   LUT, svf_p.contiguous(), Vt.contiguous(), out, out_lo, M, D, K, CH, K // CH], shared=SH_B)
                oi = (out.to(torch.int64).sum(0) * 256 + out_lo.to(torch.int64).sum(0)).double()
                # normalizacion: suma entera de los pesos (sin la escala de V)
                w = LUT.to(torch.int64)[idx].masked_fill(~keep, 0)
                sw = w.sum(1).clamp_min(1).double()
                o = (oi * svmax.double() / sw[:, None]).float()
                frac = keep.float().sum(1) / (posh + 1).float()
                resultados.setdefault(nombre, [[], []])
                resultados[nombre][0].append(o.view(B, G, D))
                resultados[nombre][1].append(frac.mean().item())

            if bs == bloques[0]:
                evaluar("denso (todos los bloques)", torch.ones_like(valido))
            for dl in deltas:
                umbral = (dl / alpha).to(torch.int64)[:, None]      # delta en unidades de z'
                evaluar(f"b{bs} d{dl} max exacto por bloque", zmax_b >= zmax[:, None] - umbral)
                evaluar(f"b{bs} d{dl} Quest + pseudo-max", cota_p >= pseudo[:, None] - umbral)
            for p in tops:
                # masa exacta por bloque (oraculo float)
                lg = (zp_pad.double() * alpha.double()[:, None, None])
                lg = lg.masked_fill(zp_pad == PAD, float("-inf"))
                masa = torch.logsumexp(lg, -1)
                prob = (masa - torch.logsumexp(masa, 1, keepdim=True)).exp()
                orden = prob.argsort(1, descending=True)
                acum = prob.gather(1, orden).cumsum(1)
                n_keep = (acum < p).sum(1) + 1
                sel = torch.zeros_like(valido)
                rango = torch.arange(nb, device=dev)[None, :]
                sel.scatter_(1, orden, rango < n_keep[:, None])
                evaluar(f"b{bs} top-p {p} (oraculo)", sel)
    fila = {}
    for nombre, (os_, fr) in resultados.items():
        o = torch.cat(os_, 1).view(B, HQ, D)
        err = 100 * ((o - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-6)).mean().item()
        fila[nombre] = (err, 100 * sum(fr) / len(fr))
    return fila


if __name__ == "__main__":
    bloques = [128, 832, 1616]
    deltas = [5, 7, 10]
    tops = [0.98, 0.995]
    for capa in [int(c) for c in (sys.argv[1:] or ["3", "35"])]:
        t0 = time.time()
        fila = correr_capa(capa, bloques, deltas, tops)
        print(f"\ncapa {capa} ({time.time()-t0:.0f}s)", flush=True)
        for nombre, (err, fr) in fila.items():
            print(f"  {nombre:40s} error salida {err:7.3f}%   keys leidas {fr:5.1f}%", flush=True)
