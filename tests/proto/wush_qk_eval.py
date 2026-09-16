"""WUSH (arXiv 2512.00956) aplicado a q.k de atencion con q/k reales (PN123).

Para cada cabeza KV y bloque de d dims, con M_q = E[q q^T] y M_k = E[k k^T]
(calibradas en el tramo 1, evaluadas en tramos 2-3):
    Q' Q'^T = damp(M_q),  K' K'^T = damp(M_k),  Q'^T K' = U S V^T
    T_q = H S^-1/2 U^T Q'^T      T_k = H S^-1/2 V^T K'^T      (T_q^T T_k = I)
=> (T_q q) . (T_k k) = q . k exacto; se cuantiza int4/int8 en el espacio transformado.
"""
import math, sys, torch

dev = "cpu"
D, HKV, HQ = 256, 2, 12
G = HQ // HKV
DIR = "/kv"
torch.manual_seed(0)


def cargar(capa):
    qs, ks, vs = [], [], []
    for i in (1, 2, 3):
        d = torch.load(f"{DIR}/capa{capa:02d}_{i}.pt")
        qs.append(d["q"]); ks.append(d["k"]); vs.append(d["v"])
    n1 = qs[0].numel() // (HQ * D)
    return (torch.cat(qs).view(-1, HQ, D).float(), torch.cat(ks).view(-1, HKV, D).float(),
            torch.cat(vs).view(-1, HKV, D).float(), n1)


def hadamard(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


def qn(x, bits, g):
    m = (1 << (bits - 1)) - 1
    sh = x.shape
    y = x.reshape(*sh[:-1], sh[-1] // g, g)
    s = y.abs().amax(-1, keepdim=True).clamp_min(1e-8) / m
    return ((y / s).round().clamp(-m, m) * s).reshape(sh)


def damp(M, a=0.01):
    return M + a * M.diagonal().mean() * torch.eye(M.shape[0], dtype=M.dtype)


def wush(qc, kc, d):
    """qc [N, d], kc [N, d] -> (T_q, T_k) [d, d] float64."""
    Mq = (qc.double().T @ qc.double()) / len(qc)
    Mk = (kc.double().T @ kc.double()) / len(kc)
    Qp = torch.linalg.cholesky(damp(Mq)); Kp = torch.linalg.cholesky(damp(Mk))
    U, S, Vh = torch.linalg.svd(Qp.T @ Kp)
    H = hadamard(d).double(); Sm = torch.diag(S.rsqrt())
    return H @ Sm @ U.T @ Qp.T, H @ Sm @ Vh @ Kp.T


def construir(q, k, n1, d, modo):
    """Transformaciones por (cabeza KV, bloque): listas de (Tq, Tk)."""
    nb = D // d
    Ts = {}
    for h in range(HKV):
        for b in range(nb):
            sl = slice(b * d, (b + 1) * d)
            qc = q[:n1, h * G:(h + 1) * G, sl].reshape(-1, d)
            kc = k[:n1, h, sl]
            if modo == "wush":
                Ts[h, b] = [t.float() for t in wush(qc, kc, d)]
            elif modo == "wush_x":      # asignacion cruzada: q usa T construida con K', k la de Q'
                Ts[h, b] = [t.float() for t in reversed(wush(qc, kc, d))]
            else:
                H = hadamard(d); Ts[h, b] = (H, H)
    return Ts


def aplicar(x, Ts, d, lado, heads_por_kv):
    out = torch.empty_like(x)
    for (h, b), (Tq, Tk) in Ts.items():
        T = Tq if lado == "q" else Tk
        hs = slice(h * heads_por_kv, (h + 1) * heads_por_kv)
        sl = slice(b * d, (b + 1) * d)
        out[:, hs, sl] = x[:, hs, sl] @ T.T
    return out


def error_logits(q, k, pos, fq, fk):
    """MSE relativa de q.k sobre pares al azar (el objetivo que WUSH minimiza)."""
    g = torch.Generator().manual_seed(1)
    tk = torch.randint(0, len(k), (4096,), generator=g)
    qq = q[pos].view(len(pos), HKV, G, D); kk = k[tk]
    ref = torch.einsum("bkgd,tkd->bkgt", qq, kk)
    aprox = torch.einsum("bkgd,tkd->bkgt", fq(q[pos]).view(len(pos), HKV, G, D), fk(kk))
    return 100 * ((aprox - ref).norm() / ref.norm()).item()


def error(q, k, v, pos, fq, fk):
    tot = 0.0
    kk = fk(k)
    for lo in range(0, len(pos), 8):
        p = pos[lo:lo + 8]; T = int(p.max()) + 1
        qq_ref = q[p].view(len(p), HKV, G, D)
        qq = fq(q[p]).view(len(p), HKV, G, D)
        mask = torch.arange(T)[None, :] > p[:, None]
        o = []
        for qx, kx in ((qq_ref, k[:T]), (qq, kk[:T])):
            sc = torch.einsum("bkgd,tkd->bkgt", qx, kx) / math.sqrt(D)
            w = sc.masked_fill(mask[:, None, None, :], float("-inf")).softmax(-1)
            o.append(torch.einsum("bkgt,tkd->bkgd", w, v[:T]).reshape(len(p), HQ, D))
        tot += ((o[1] - o[0]).norm(dim=-1) / o[0].norm(dim=-1).clamp_min(1e-6)).sum().item()
    return 100 * tot / (len(pos) * HQ)


if __name__ == "__main__":
  for capa in [int(c) for c in (sys.argv[1:] or ["3", "35"])]:
      q, k, v, n1 = cargar(capa)
      N = k.shape[0]
      pos = torch.linspace(n1 + 512, N - 1, 96).long()        # evaluacion fuera de la calibracion
      print(f"\ncapa {capa}: N={N} calibracion={n1}", flush=True)
      for bits, g in ((4, 64), (8, 256)):
          fila = []
          for d in (256, 64):
              if g > d:
                  continue
              for modo in ("hadamard", "wush", "wush_x"):
                  Ts = construir(q, k, n1, d, modo)
                  # q.k en el espacio transformado == q.k original, asi que el softmax
                  # se calcula directo sobre los cuantizados transformados.
                  fq = lambda x, Ts=Ts, d=d: qn(aplicar(x, Ts, d, "q", G), bits, g)
                  fk = lambda x, Ts=Ts, d=d: qn(aplicar(x, Ts, d, "k", 1), bits, g)
                  fila.append((d, modo, fq, fk))
          for d, modo, fq, fk in fila:
              print(f"  int{bits} grupo {g:3d} | bloque {d:3d} {modo:8s}: q.k {error_logits(q, k, pos, fq, fk):.2f}%  error salida {error(q, k, v, pos, fq, fk):.3f}%", flush=True)
