"""RheoSampling en el arbol de borrador: prueba ESTADISTICA de que no cambia la distribucion.

Target de juguete (Markov: los logits de una fila dependen de la profundidad y del ultimo token),
borrador de juguete con otra distribucion, 400k pedidos en un solo lanzamiento. Se arma el arbol
con ``construir_dfs_kernel(rheo=...)``, se verifica con ``preparar_rheo`` + ``aceptar_kernel``, y
lo que el paso no emitio se completa muestreando del target (lo que haria el paso siguiente).
La conjunta de los 3 primeros tokens tiene que ser la del target: se compara su distancia de
variacion total contra la de un muestreo DIRECTO del target con el mismo N (el piso de ruido).

Control negativo: NEG=1 verifica al hijo muestreado como si fuera determinista (sin p/q): tiene
que dar una distancia claramente mayor que el piso.
"""
import os, torch
import vllm._genesis.arbol_borrador as ab
NEG = int(os.environ.get("NEG", "0"))
dev = "cuda"; torch.manual_seed(1)
V, K, S, N, R = 8, 4, 3, 4, 400_000
T = N + 1
TEMP_Q = float(os.environ.get("TEMP_Q", "0.7"))
cand1 = torch.stack([torch.randperm(V)[:K] for _ in range(S)]).to(dev)            # [S, K]
SC = (torch.randn(S, K, K) * 1.5).to(dev)
LT = (torch.randn(S + 1, V, V) * 1.2).to(dev)                                      # [prof, prev, V]
cand = cand1[None].expand(R, S, K).contiguous()
sc = SC[None].expand(R, S, K, K).contiguous()
o_tok = torch.zeros(R, N, dtype=torch.int64, device=dev)
o_pad = torch.full((R, T), -1, dtype=torch.int32, device=dev)
o_prof = torch.zeros(R, T, dtype=torch.int64, device=dev)
o_bits = torch.zeros(R, T, dtype=torch.int32, device=dev)
temp = torch.full((R,), TEMP_Q, dtype=torch.float32, device=dev)
seed = torch.arange(R, dtype=torch.int64, device=dev) * 7919 + 13
pos = torch.full((R,), 5, dtype=torch.int64, device=dev)
o_qres = torch.zeros(R, T, K, dtype=torch.float32, device=dev)
o_cand = torch.zeros(R, T, K, dtype=torch.int64, device=dev)
o_hs = torch.full((R, T), -1, dtype=torch.int32, device=dev)
o_qs = torch.zeros(R, T, dtype=torch.float32, device=dev)


def gumbel(lg):
    u = torch.rand_like(lg, dtype=torch.float32).clamp_(1e-20, 1.0)
    return (lg.float() - torch.log(-torch.log(u))).argmax(-1)


def correr(con_rheo):
    rh = (temp, seed, pos, o_qres, o_cand, o_hs, o_qs) if con_rheo else None
    o_hs.fill_(-1)
    ab.construir_dfs_kernel(cand, sc, o_tok, o_pad, o_prof, N, o_bits, rheo=rh)
    tok = torch.cat([torch.zeros(R, 1, dtype=torch.int64, device=dev), o_tok], 1)  # ancla = token 0
    logits = LT[o_prof.flatten(), tok.flatten()].contiguous()                       # [R*T, V]
    cu = (torch.arange(R + 1, device=dev) * T).to(torch.int32)
    fila = torch.arange(R * T, device=dev) // T
    y_p = gumbel(logits)
    todos = torch.ones(R, T, dtype=torch.bool, device=dev)
    if con_rheo:
        hs = o_hs.clone()
        razon, y_r = ab.preparar_rheo(logits, tok.flatten(), cu, fila, hs.flatten(), o_qs.flatten(),
                                      o_cand.view(R * T, K), o_qres.view(R * T, K), gumbel)
        if NEG:
            razon = torch.full_like(razon, 1e9); y_r = y_p       # "aceptar siempre al muestreado"
        pos_f = (torch.arange(R * T, device=dev) % T).to(torch.int64)
        out = ab.aceptar_kernel(tok.flatten(), y_p, cu, o_pad, todos, T, N,
                                rheo=(hs, razon, y_r, seed, pos_f))
    else:
        out = ab.aceptar_kernel(tok.flatten(), y_p, cu, o_pad, todos, T, N)
    sampled, nacc = out[0], out[1].long()
    sec = torch.zeros(R, 3, dtype=torch.int64, device=dev)
    prev = torch.zeros(R, dtype=torch.int64, device=dev)
    for j in range(3):
        fresco = gumbel(LT[j, prev])
        sec[:, j] = torch.where(j < nacc, sampled[:, j], fresco)
        prev = sec[:, j]
    return sec, nacc.float().mean().item(), (o_hs >= 0).float().sum(1).mean().item()


def tv(sec):
    h = torch.bincount(sec[:, 0] * V * V + sec[:, 1] * V + sec[:, 2], minlength=V ** 3).float() / sec.shape[0]
    p0 = LT[0, 0].softmax(-1); p1 = LT[1].softmax(-1); p2 = LT[2].softmax(-1)
    ex = (p0[:, None, None] * p1[:, :, None] * p2[None, :, :]).flatten()
    return 0.5 * (h - ex).abs().sum().item()


directo = torch.zeros(R, 3, dtype=torch.int64, device=dev); prev = torch.zeros(R, dtype=torch.int64, device=dev)
for j in range(3):
    directo[:, j] = gumbel(LT[j, prev]); prev = directo[:, j]
print(f"piso de ruido (muestreo directo del target, N={R}): TV = {tv(directo):.5f}")
for nombre, cr in (("arbol determinista (one-hot)", False), ("arbol RheoSampling" + (" [NEG: sin p/q]" if NEG else ""), True)):
    sec, acc, nmu = correr(cr)
    print(f"{nombre:42s}: TV = {tv(sec):.5f} | emitidos por paso {acc:.3f} | hijos muestreados en el arbol {nmu:.2f}")
