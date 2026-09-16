"""SK-18 A0-ter — estimadores para ELEGIR tiles de keys antes del calculo exacto.

Todo el calculo pesado con kernels PTX:
  scores exactos           SK-18a (mma s8)
  miniaturas int4 q/k      SK-18c (mma s4, nibbles)
  PCA r dims (int8)        SK-18a con K=r (mma s8)
  atencion entera final    SK-18b (softmax entero + sum w*v)
Estimadores (tiles de 128 keys):
  E1  q4.k4 (int4 por token-cabeza de q y k rotadas)
  E2  PCA r=128 / r=64 de k (calibrada en el tramo 1), q proyectada; int8
Seleccion:
  simple       sobre el score estimado
  cota-peor    [s - e, s + e], e = cota del peor caso del error de redondeo
                (E1: sum|q^|sk/2 + sum|k^|sq/2 + D sq sk/4; E2: + |q_perp||k_perp|)
  cota-3sigma  e = 3 * desvio del error de redondeo (uniforme: paso^2/12)
Criterios: delta 7/10 (bloque si cota_sup >= max(cota_inf) - delta) y top-p
certificado (masa inferior de lo elegido >= p * masa superior total).
PISA: los tiles descartados se aproximan con su key y valor medio (un termino
por tile) en vez de tirarlos.
"""
import math, sys, time, torch
sys.argv = sys.argv[:1] + sys.argv[1:]
import sk18_a0bis_ptx as A
from vllm._genesis.kernels.ptx_lab import Kernel

dev = "cuda"
D, HKV, HQ, G = 256, 2, 12, 6
BS = 128
PAD = A.PAD
SH_A = 2 * (256 * 128 + 64 * 2 * 128)
k4ker = Kernel("sk18a_qk_i32.cu", "sk18a_qk_i32", defs=["-DS4=1"], warps=8)


def nib(x):
    u = (x & 0xF).to(torch.uint8)
    return (u[:, 0::2] | (u[:, 1::2] << 4)).view(torch.int8).contiguous()


def qk_s4(A4, B4):
    M, N = A4.shape[0], B4.shape[0]
    out = torch.empty(M, N, dtype=torch.int32, device=dev)
    k4ker.lanzar(((N + 63) // 64, (M + 255) // 256), [nib(A4), nib(B4), out, M, N, 128], shared=SH_A)
    return out


def qk_r(Ai, Bi):
    """int8 [M,r<=128] x int8 [N,r] con SK-18a K=128 (relleno con ceros: exacto)."""
    M, N, r = Ai.shape[0], Bi.shape[0], Ai.shape[1]
    Ap = torch.zeros(M, 128, dtype=torch.int8, device=dev); Ap[:, :r] = Ai
    Bp = torch.zeros(N, 128, dtype=torch.int8, device=dev); Bp[:, :r] = Bi
    out = torch.empty(M, N, dtype=torch.int32, device=dev)
    A.k18a.lanzar(((N + 63) // 64, (M + 255) // 256), [Ap, Bp, out, M, N, 128], shared=SH_A)
    return out


def q4(x):
    s = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 7
    return (x / s).round().clamp(-7, 7).to(torch.int8), s.squeeze(-1)


def q8(x):
    s = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
    return (x / s).round().clamp(-127, 127).to(torch.int8), s.squeeze(-1)


def salida_entera(ctx, keep, pisa=False):
    """Atencion entera de SK-18b sobre las keys de `keep` [M, N]; opcional PISA."""
    zp, mq, svf_p, Vt, svmax, K, M, N = (ctx[k] for k in ("zp", "mq", "svf_p", "Vt", "svmax", "K", "M", "N"))
    CH = 1024
    Sz = zp.masked_fill(~keep, PAD)
    zm = Sz.amax(1)
    d = (zm[:, None] - Sz).clamp(min=0)
    idx = ((d * mq[:, None].to(torch.int64) + 32768) >> 16).clamp(max=1023)
    Szp = torch.full((M, K), PAD, dtype=torch.int32, device=dev)
    Szp[:, :N] = Sz.clamp(min=PAD).to(torch.int32)
    out = torch.zeros(K // CH, M, D, dtype=torch.int32, device=dev)
    out_lo = torch.zeros_like(out)
    grid = ((D + 63) // 64, ((M + 63) // 64) * (K // CH))
    A.k18b.lanzar(grid, [Szp.contiguous(), zm.to(torch.int32).contiguous(), mq.contiguous(),
                         torch.zeros(M, dtype=torch.int32, device=dev), A.LUT, svf_p, Vt, out, out_lo,
                         M, D, K, CH, K // CH], shared=A.SH_B)
    oi = (out.to(torch.int64).sum(0) * 256 + out_lo.to(torch.int64).sum(0)).double()
    w = A.LUT.to(torch.int64)[idx].masked_fill(~keep, 0)
    sw = w.sum(1).double()
    num = oi * svmax.double()
    if pisa:
        # tiles descartados: masa ~ n * exp(logit(q.k_media) - logit_max) y valor medio
        tiles_desc = ctx["tile_valido"] & ~ctx["tile_keep"]
        lg_med = ctx["lg_media"]                                   # [M, nb] logit real de la key media
        lmax = (zm.double() * ctx["alpha"].double())[:, None]
        masa = (tiles_desc.double() * ctx["n_tile"][None, :] * torch.exp(lg_med - lmax) * 32639.0)
        num = num + masa @ ctx["v_media"].double()                   # [M, D]
        sw = sw + masa.sum(1)
    return (num / sw.clamp_min(1)[:, None]).float()


def correr_capa(capa):
    q, k, v = A.cargar(capa)
    N = k.shape[0]
    pos = torch.linspace(8000, N - 1, 48, device=dev).long()
    ref = A.referencia(q, k, v, pos)
    B = len(pos)
    n1 = 7488
    qr = A.rotar_ptx(q[pos]); kr = A.rotar_ptx(k)
    q8r, sq = A.q8_ptx(qr); k8r, sk = A.q8_ptx(kr); v8, sv = A.q8_ptx(v)
    res = {}
    nb = (N + BS - 1) // BS
    for h in range(HKV):
        M = B * G
        posh = pos.repeat_interleave(G)
        Qh8 = q8r[:, h * G:(h + 1) * G].reshape(M, D); sqh = sq[:, h * G:(h + 1) * G].reshape(M)
        qf = qr[:, h * G:(h + 1) * G].reshape(M, D)                  # q rotada float
        kf = kr[:, h]                                                 # k rotada float
        # exactos
        z = A.qk_ptx(Qh8, k8r[:, h])
        skh = sk[:, h]; skmax = skh.max()
        skf = torch.round(skh / skmax * 32767).to(torch.int64)
        zp = ((z.to(torch.int64) >> 8) * skf[None, :]) >> 11
        causal = torch.arange(N, device=dev)[None, :] > posh[:, None]
        zp = zp.masked_fill(causal, PAD)
        alpha = 256.0 * skmax * sqh / math.sqrt(D) / 16.0
        mq = torch.round(alpha * 1023 / A.C_RECORTE * 65536).clamp(max=(1 << 31) - 1).to(torch.int32)
        svh = sv[:, h]; svmax = svh.max()
        K = ((N + 1023) // 1024) * 1024
        Vt = torch.zeros(D, K, dtype=torch.int8, device=dev); Vt[:, :N] = v8[:, h].t()
        svf_p = torch.zeros(K, dtype=torch.int16, device=dev); svf_p[:N] = torch.round(svh / svmax * 32767).to(torch.int16)
        lg_exacto = (zp.double() * alpha.double()[:, None]).masked_fill(causal, -1e9)
        tile_valido = (torch.arange(nb, device=dev)[None, :] * BS) <= posh[:, None]
        tile_ult = (posh // BS)
        # medias por tile (PISA): key y valor medios en float
        kpad = torch.nn.functional.pad(k[:, h], (0, 0, 0, nb * BS - N)).view(nb, BS, D)
        vpad = torch.nn.functional.pad(v[:, h], (0, 0, 0, nb * BS - N)).view(nb, BS, D)
        n_tile = torch.full((nb,), float(BS), device=dev, dtype=torch.float64); n_tile[-1] = N - (nb - 1) * BS
        k_media = kpad.sum(1) / n_tile[:, None].float(); v_media = (vpad.sum(1) / n_tile[:, None].float())
        q_orig = q[pos][:, h * G:(h + 1) * G].reshape(M, D)
        lg_media = (q_orig.double() @ k_media.double().t()) / math.sqrt(D)
        ctx = dict(zp=zp, mq=mq, svf_p=svf_p, Vt=Vt, svmax=svmax, K=K, M=M, N=N, alpha=alpha,
                   tile_valido=tile_valido, n_tile=n_tile, lg_media=lg_media, v_media=v_media)

        def tiles(x, red):
            xp = torch.nn.functional.pad(x, (0, nb * BS - N), value=-1e9 if red == "max" else 0.0).view(M, nb, BS)
            return xp.amax(-1) if red == "max" else xp

        def evaluar(nombre, keep_t):
            keep_t = (keep_t | (torch.arange(nb, device=dev)[None, :] == tile_ult[:, None]) |
                      (torch.arange(nb, device=dev)[None, :] == 0)) & tile_valido
            keep = keep_t.repeat_interleave(BS, dim=1)[:, :N] & ~causal
            ctx["tile_keep"] = keep_t
            for pisa in (False, True):
                o = salida_entera(ctx, keep, pisa)
                nom = nombre + (" +PISA" if pisa else "")
                res.setdefault(nom, [[], []])
                res[nom][0].append(o.view(B, G, D))
                res[nom][1].append((keep.float().sum(1) / (posh + 1).float()).mean().item())

        # ---------- oraculo
        tmax_ex = tiles(lg_exacto, "max")
        lmax = tmax_ex.amax(1, keepdim=True)
        evaluar("denso", torch.ones_like(tile_valido))
        for dl in (7, 10):
            evaluar(f"oraculo d{dl}", tmax_ex >= lmax - dl)

        # ---------- estimadores: devuelven (estimado, error) por key en logits reales
        estimadores = {}
        # E1 int4
        q4i, q4s = q4(qf); k4i, k4s = q4(kf)
        s4 = qk_s4(q4i, k4i).double() * q4s.double()[:, None] * k4s.double()[None, :] / math.sqrt(D)
        qh = (q4i.double() * q4s.double()[:, None]); kh = (k4i.double() * k4s.double()[:, None])
        e_peor = (qh.abs().sum(1)[:, None] * k4s.double()[None, :] / 2 + kh.abs().sum(1)[None, :] * q4s.double()[:, None] / 2
                  + D * q4s.double()[:, None] * k4s.double()[None, :] / 4) / math.sqrt(D)
        e_3s = 3 * torch.sqrt((qh.square().sum(1)[:, None] * k4s.double().square()[None, :] +
                               kh.square().sum(1)[None, :] * q4s.double().square()[:, None]) / 12) / math.sqrt(D)
        estimadores["E1 int4"] = (s4, e_peor, e_3s)
        # E2 PCA
        kc = kf[:n1].double()
        cov = kc.t() @ kc / n1
        evals, evecs = torch.linalg.eigh(cov)
        P = evecs.flip(1)                                          # componentes de mayor a menor
        for r in (128, 64):
            Pr = P[:, :r].float()
            kp = kf @ Pr; qp = qf @ Pr
            kperp = (kf - kp @ Pr.t()).norm(dim=1).double(); qperp = (qf - qp @ Pr.t()).norm(dim=1).double()
            kpi, kps = q8(kp); qpi, qps = q8(qp)
            zr = qk_r(qpi, kpi).double()
            sr = zr * qps.double()[:, None] * kps.double()[None, :] / math.sqrt(D)
            qph = qpi.double() * qps.double()[:, None]; kph = kpi.double() * kps.double()[:, None]
            e_q = (qph.abs().sum(1)[:, None] * kps.double()[None, :] / 2 + kph.abs().sum(1)[None, :] * qps.double()[:, None] / 2) / math.sqrt(D)
            e_cs = qperp[:, None] * kperp[None, :] / math.sqrt(D)
            e_3s = 3 * torch.sqrt((qph.square().sum(1)[:, None] * kps.double().square()[None, :] +
                                   kph.square().sum(1)[None, :] * qps.double().square()[:, None]) / 12) / math.sqrt(D) + e_cs * 0.3
            estimadores[f"E2 PCA r{r}"] = (sr, e_q + e_cs, e_3s)

        for nom_e, (s_est, e_peor, e_3s) in estimadores.items():
            s_est = s_est.masked_fill(causal, -1e9)
            for modo, e in (("simple", None), ("cota-peor", e_peor), ("cota-3sigma", e_3s)):
                if e is None:
                    up = lo = s_est
                else:
                    up = (s_est + e).masked_fill(causal, -1e9); lo = (s_est - e).masked_fill(causal, -1e9)
                tup = tiles(up, "max"); Lmax = tiles(lo, "max").amax(1, keepdim=True)
                for dl in (7, 10):
                    evaluar(f"{nom_e} {modo} d{dl}", tup >= Lmax - dl)
                # top-p certificado sobre masas por tile
                lse = lambda x: torch.logsumexp(tiles(x, "pad"), -1)
                m_up = lse(up); m_lo = lse(lo)
                for p in (0.99, 0.995):
                    orden = m_up.argsort(1, descending=True)
                    up_ord = m_up.gather(1, orden).exp(); lo_ord = m_lo.gather(1, orden).exp()
                    tot_up = up_ord.sum(1, keepdim=True)
                    cub = lo_ord.cumsum(1) + (tot_up - up_ord.cumsum(1))       # masa inferior elegida + superior del resto
                    ok = lo_ord.cumsum(1) >= p * cub
                    n_keep = torch.where(ok.any(1), ok.float().argmax(1) + 1, torch.full_like(ok[:, 0], nb, dtype=torch.long))
                    sel = torch.zeros_like(tile_valido)
                    sel.scatter_(1, orden, torch.arange(nb, device=dev)[None, :] < n_keep[:, None])
                    evaluar(f"{nom_e} {modo} top-p{p}", sel)
    fila = {}
    for nom, (os_, fr) in res.items():
        o = torch.cat(os_, 1).view(B, HQ, D)
        err = 100 * ((o - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-6)).mean().item()
        fila[nom] = (err, 100 * sum(fr) / len(fr))
    return fila


if __name__ == "__main__":
    for capa in [int(c) for c in (sys.argv[1:] or ["3", "35"])]:
        t0 = time.time()
        fila = correr_capa(capa)
        print(f"\ncapa {capa} ({time.time()-t0:.0f}s)", flush=True)
        for nom, (err, fr) in fila.items():
            print(f"  {nom:40s} error {err:6.3f}%   keys leidas {fr:5.1f}%", flush=True)
