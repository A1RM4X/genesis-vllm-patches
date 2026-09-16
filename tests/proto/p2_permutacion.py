"""P2 — permutacion de canales en el MLP con pesos reales (Ar4ikov AWQ).

La permutacion pi de las neuronas intermedias (filas de gate/up, columnas de down)
es EXACTA: down[:, pi] @ (silu(gate[pi] x) * up[pi] x) == down @ (silu(gate x)*up x).
Se usa gratis para:
  (a) agrupar columnas de down_proj con magnitudes parecidas antes de cuantizar
      int4 por grupos de 64 (el grupo del mma k64);
  (b) alinear la mascara 2:4 de down_proj para retener mas magnitud (ASP).
"""
import json, glob, os, struct, sys, time
import torch

D = glob.glob("/m/hub/models--Ar4ikov--Qwen3.8-27B-Uncensored-AWQ-W4A16-ASYM/snapshots/*/")[0]
torch.manual_seed(0)


def tensores(nombres):
    out = {}
    for f in sorted(glob.glob(D + "*.safetensors")):
        with open(os.path.realpath(f), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
            base = 8 + n
            for k, v in h.items():
                if k in nombres:
                    s, e = v["data_offsets"]
                    fh.seek(base + s)
                    buf = bytearray(fh.read(e - s))
                    dt = {"I32": torch.int32, "BF16": torch.bfloat16, "I64": torch.int64}[v["dtype"]]
                    out[k] = torch.frombuffer(buf, dtype=dt).reshape(v["shape"]).clone()
    return out


def desempacar8(p, total, dim):
    """int32 con 8 nibbles (LSB primero) -> uint4 a lo largo de `dim`."""
    sh = torch.arange(0, 32, 4, dtype=torch.int64)
    x = (p.to(torch.int64).unsqueeze(-1) >> sh) & 0xF
    if dim == 1:
        return x.reshape(p.shape[0], -1)[:, :total]
    return x.permute(1, 0, 2).reshape(p.shape[1], -1)[:, :total].t()   # dim 0: por filas


def dequant(pref):
    t = tensores({pref + s for s in (".weight_packed", ".weight_scale", ".weight_zero_point", ".weight_shape")})
    out_f, in_f = t[pref + ".weight_shape"].tolist()
    q = desempacar8(t[pref + ".weight_packed"], in_f, 1)                      # [out, in]
    sc = t[pref + ".weight_scale"].float()                                   # [out, G]
    G = sc.shape[1]
    zp = desempacar8(t[pref + ".weight_zero_point"], out_f, 0)                # [out, G]
    g = in_f // G
    idx = torch.arange(in_f) // g
    return ((q - zp[:, idx]).float() * sc[:, idx])


def err_int4_g(W, g=64):
    K = W.shape[1]; pad = (-K) % g
    y = torch.nn.functional.pad(W, (0, pad)).reshape(W.shape[0], -1, g)
    s = y.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 7
    q = ((y / s).round().clamp(-7, 7) * s).reshape(W.shape[0], -1)[:, :K]
    return ((q - W).norm() / W.norm()).item()


def retenido_24(W, perm):
    Wp = W[:, perm].abs()
    K = Wp.shape[1] - Wp.shape[1] % 4
    g = Wp[:, :K].reshape(Wp.shape[0], -1, 4)
    return (g.topk(2, -1).values.sum() / g.sum()).item()


def busqueda_24(W, iters=4000):
    """Intercambios entre grupos de 4 columnas (estilo ASP): acepta si sube lo retenido."""
    A = W.abs()
    K = A.shape[1] - A.shape[1] % 4
    perm = torch.arange(A.shape[1])
    cols = A[:, perm[:K]].reshape(A.shape[0], -1, 4).permute(1, 0, 2).contiguous()   # [grupos, filas, 4]
    val = cols.topk(2, -1).values.sum((1, 2))
    ng = cols.shape[0]
    for _ in range(iters):
        i, j = torch.randint(0, ng, (2,)).tolist()
        if i == j:
            continue
        a, b = torch.randint(0, 4, (2,)).tolist()
        ci, cj = cols[i].clone(), cols[j].clone()
        ci[:, a], cj[:, b] = cols[j][:, b], cols[i][:, a]
        vi, vj = ci.topk(2, -1).values.sum(), cj.topk(2, -1).values.sum()
        if vi + vj > val[i] + val[j]:
            cols[i], cols[j], val[i], val[j] = ci, cj, vi, vj
            pi, pj = i * 4 + a, j * 4 + b
            perm[pi], perm[pj] = perm[pj].item(), perm[pi].item()
    return perm


for capa in (int(x) for x in (sys.argv[1:] or ["10", "40"])):
    t0 = time.time()
    p = f"model.language_model.layers.{capa}.mlp."
    Wg, Wu, Wd = dequant(p + "gate_proj"), dequant(p + "up_proj"), dequant(p + "down_proj")
    N = Wg.shape[0]
    # 1) exactitud
    x = torch.randn(4, Wg.shape[1], dtype=torch.float64)
    f = lambda g, u, d: (torch.nn.functional.silu(x @ g.double().t()) * (x @ u.double().t())) @ d.double().t()
    pi = torch.randperm(N)
    ref = f(Wg, Wu, Wd); per = f(Wg[pi], Wu[pi], Wd[:, pi])
    print(f"capa {capa}: permutacion exacta -> max dif {(ref - per).abs().max().item():.2e} (rel {((ref-per).norm()/ref.norm()).item():.1e})", flush=True)
    # 2) int4 g64 sobre columnas de down
    ident = torch.arange(N)
    por_norma = torch.argsort(Wd.norm(dim=0))
    por_max = torch.argsort(Wd.abs().amax(dim=0))
    print(f"   down int4 g64 err: identidad {100*err_int4_g(Wd):.2f}%  orden por norma {100*err_int4_g(Wd[:, por_norma]):.2f}%  "
          f"orden por max {100*err_int4_g(Wd[:, por_max]):.2f}%", flush=True)
    print(f"   gate int4 g64 err (sin permutar, K=hidden): {100*err_int4_g(Wg):.2f}%  up {100*err_int4_g(Wu):.2f}%", flush=True)
    # 3) 2:4 en down (columnas = neuronas intermedias, permutables gratis)
    print(f"   down 2:4 magnitud retenida: identidad {100*retenido_24(Wd, ident):.2f}%  por norma {100*retenido_24(Wd, por_norma):.2f}%", flush=True)
    pb = busqueda_24(Wd, iters=3000)
    print(f"   down 2:4 con busqueda ASP (3000 intercambios): {100*retenido_24(Wd, pb):.2f}%   ({time.time()-t0:.0f}s)", flush=True)
