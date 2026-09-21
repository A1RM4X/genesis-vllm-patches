"""Hito 3 del arbol de borrador: GDN en arbol sobre la cinta de PN122.

Compara ``gdn_cinta.spec_update`` con GENESIS_ENABLE_ARBOL=1 contra una referencia en fp32
escrita aparte (estado por nodo = regla delta sobre el estado de SU padre), a lo largo de varios
pasos con aceptaciones al azar: asi se prueba tambien que lo que se reproduce al paso siguiente
es el CAMINO aceptado y no las primeras filas de la cinta. Y el control: con la mascara de
cadena la salida tiene que ser identica bit a bit a la del kernel de produccion.

Correr adentro del contenedor:  GENESIS_ENABLE_ARBOL=1 GENESIS_ENABLE_PN122_GDN_CINTA=1 python3 gdn_arbol.py
"""
import os, sys, types, torch
os.environ["GENESIS_ENABLE_ARBOL"] = "1"
os.environ["GENESIS_ENABLE_PN122_GDN_CINTA"] = "1"
import vllm._genesis.gdn_cinta as g
import vllm._genesis.arbol_borrador as ab

torch.manual_seed(3)
dev, dt = "cuda", torch.float16
H, HV, K, V, SPEC, N = 8, 24, 128, 128, 8, 4
T = SPEC + 1
PASOS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
NEG = int(os.environ.get("NEG", "0"))
S = 1 + N
raw = torch.randn(S, HV * V * K + 1000, device=dev, dtype=dt) * 0.05
h = raw[:, :HV * V * K].view(S, HV, V, K)
A_log = torch.randn(HV, device=dev) * 0.5
dt_bias = torch.randn(HV, device=dev) * 0.5
cols = torch.arange(1, S, device=dev, dtype=torch.int32).view(N, 1)
g._init_slots(8, dev)
layer = types.SimpleNamespace(tp_size=2, num_k_heads=2 * H, num_v_heads=2 * HV, head_k_dim=K,
                              head_v_dim=V, num_spec=SPEC, prefix="t")
g.enlazar(layer, dev)
slots = torch.tensor([3, 7, 1, 5], device=dev, dtype=torch.int32)
cu = torch.arange(0, (N + 1) * T, T, device=dev, dtype=torch.int32)
anc_buf = torch.zeros(N * T, dtype=torch.int32, device=dev)
g.fijar_ancestros(anc_buf)
camino = g.camino_gpu()
assert camino is not None and camino.shape[1] == SPEC


def delta(Sx, kk, vv, gg, bb):
    Sx = Sx * torch.exp(gg)[:, None, None]
    d = (vv - torch.einsum("hvk,hk->hv", Sx, kk)) * bb[:, None]
    return Sx + d[:, :, None] * kk[:, None, :]


def entradas(q, k, v, a, b, i):
    kk = k[0, i].float(); qq = q[0, i].float()
    kk = kk * torch.rsqrt((kk * kk).sum(-1, keepdim=True) + 1e-6)
    qq = qq * torch.rsqrt((qq * qq).sum(-1, keepdim=True) + 1e-6) * K ** -0.5
    rep = HV // H
    x = a[i].float() + dt_bias
    sp = torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)
    return (qq.repeat_interleave(rep, 0), kk.repeat_interleave(rep, 0), v[0, i].float(),
            -torch.exp(A_log) * sp, torch.sigmoid(b[i].float()))


def arbol_al_azar(n, gen, dfs):
    """Padres de [ancla]+n nodos. dfs=False deja un orden topologico cualquiera."""
    pad = torch.tensor([[int(torch.randint(-1, i, (1,), generator=gen)) for i in range(n)]])
    if dfs:
        prof = torch.zeros_like(pad)
        for i in range(n):
            prof[0, i] = 0 if pad[0, i] < 0 else prof[0, pad[0, i]] + 1
        _, pad, _, _, _ = ab.orden_dfs(pad.clone(), pad, prof, pad.clone())
    return ab.padres_verificacion(pad)[0]          # [T], el 0 es el ancla


gen = torch.Generator().manual_seed(11)
h_ref = h.clone()
acc = torch.ones(N, device=dev, dtype=torch.int32)
pendiente = [[] for _ in range(N)]                 # entradas aceptadas del paso anterior
peor, saltos_tot = 0.0, 0
for paso in range(PASOS):
    q = torch.randn(1, N * T, H, K, device=dev, dtype=dt); k = torch.randn_like(q)
    v = torch.randn(1, N * T, HV, V, device=dev, dtype=dt)
    a = torch.randn(N * T, HV, device=dev, dtype=dt); b = torch.randn_like(a)
    padres = []
    for n in range(N):
        p = torch.arange(-1, T - 1) if n == 0 else arbol_al_azar(SPEC, gen, dfs=(n != 3))
        padres.append(p)
        bits = ab.bits_ancestros(p[None])[0]
        if NEG == 1:      # control negativo: el kernel cree que todo es cadena
            bits = ((1 << torch.arange(T)) - 1).to(torch.int32)
        anc_buf[n * T:(n + 1) * T] = bits.to(dev)
        saltos_tot += sum(1 for t in range(2, T) if int(p[t]) != t - 1)
    o, _ = g.spec_update(layer, A_log, a, b, dt_bias, q, k, v, h, cu, cols, acc, slots)
    torch.cuda.synchronize()

    o_ref = torch.zeros(N * T, HV, V, device=dev)
    nuevo_acc, nuevo_pend = [], []
    for n in range(N):
        Sx = h_ref[1 + n].float()
        for e in pendiente[n]:
            Sx = delta(Sx, *e)
        est, ent = {}, {}
        for t in range(T):
            qq, kk, vv, gg, bb = entradas(q, k, v, a, b, n * T + t)
            ent[t] = (kk, vv, gg, bb)
            base = Sx if t == 0 else est[int(padres[n][t])]
            est[t] = delta(base, kk, vv, gg, bb)
            o_ref[n * T + t] = torch.einsum("hvk,hk->hv", est[t], qq)
            if t == 0:
                h_ref[1 + n] = est[0].to(dt)
                est[0] = h_ref[1 + n].float()      # el kernel tambien sigue... en fp32
                est[0] = delta(base, kk, vv, gg, bb)
        # aceptar un camino al azar
        hoja = int(torch.randint(0, T, (1,), generator=gen))
        cam = []
        while hoja > 0:
            cam.append(hoja); hoja = int(padres[n][hoja])
        cam.reverse()
        nuevo_acc.append(len(cam) + 1)
        nuevo_pend.append([ent[t] for t in cam])
        fila = torch.arange(SPEC, dtype=torch.int32)
        fila[:len(cam)] = torch.tensor([t - 1 for t in cam], dtype=torch.int32)
        if NEG != 2:      # control negativo 2: se reproducen las primeras filas, no el camino
            camino[int(slots[n])] = fila.to(dev)
    err = ((o[0].float() - o_ref).norm(dim=(1, 2)) / o_ref.norm(dim=(1, 2))).view(N, T)
    peor = max(peor, float(err.max()))
    print(f"paso {paso:2d} acc_prev={acc.tolist()} err rel por pedido (max sobre tokens): "
          f"{[f'{x:.1e}' for x in err.max(1).values.tolist()]}")
    acc = torch.tensor(nuevo_acc, device=dev, dtype=torch.int32)
    pendiente = nuevo_pend
print(f"PEOR error relativo contra la referencia fp32: {peor:.2e}   (saltos de rama: {saltos_tot})"
      f"   [{'FORMA CERRADA' if os.environ.get('GENESIS_ARBOL_CERRADA','1') in ('1','true') else 'secuencial'}]")

# ---- control: mascara de cadena == kernel de produccion, bit a bit ----
hA, hB = h.clone(), h.clone()
cintaA = layer._g122_cinta.clone()
anc_buf.copy_(((1 << torch.arange(T)) - 1).to(torch.int32).repeat(N).to(dev))
camino.copy_(torch.arange(SPEC, dtype=torch.int32, device=dev)[None].expand_as(camino))
acc = torch.randint(1, T + 1, (N,), device=dev, dtype=torch.int32)
oA, _ = g.spec_update(layer, A_log, a, b, dt_bias, q, k, v, hA, cu, cols, acc, slots)
layer._g122_cinta.copy_(cintaA)
g.fijar_ancestros(None)
oB, _ = g.spec_update(layer, A_log, a, b, dt_bias, q, k, v, hB, cu, cols, acc, slots)
print("cadena: salida identica =", torch.equal(oA, oB), " estado identico =", torch.equal(hA, hB))
