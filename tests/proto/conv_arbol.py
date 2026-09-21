"""Hito 4: conv causal de GDN por camino, contra una referencia fp32, en varios pasos y usando
el kernel REAL de upstream para escribir el estado. Controles: NEG=1 (sin compactar),
y la cadena (salidas == upstream, compactar no copia nada)."""
import os, sys, torch
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
import vllm._genesis.arbol_conv as ac
import vllm._genesis.arbol_borrador as ab

NEG = int(os.environ.get("NEG", "0"))
torch.manual_seed(5); gen = torch.Generator().manual_seed(9)
dev, dt = "cuda", torch.float16
DIM, W, SPEC, N, PASOS = 5120, 4, 8, 4, 12
T = SPEC + 1; SL = W - 1 + SPEC
NB = N + 3
cs_raw = torch.randn(NB, SL, DIM, device=dev, dtype=dt)          # como en vLLM: se usa transpuesto
cs = cs_raw.transpose(-1, -2)                                     # [bloques, dim, state_len]
w = torch.randn(DIM, W, device=dev, dtype=dt) * 0.3
sidx = torch.tensor([2, 5, 1, 4], device=dev, dtype=torch.int32)
cu = torch.arange(0, (N + 1) * T, T, device=dev, dtype=torch.int32)
anc3 = torch.zeros(N * T, 3, dtype=torch.int32, device=dev)
camino = torch.arange(SPEC, dtype=torch.int32, device=dev)[None].repeat(N, 1).contiguous()
fila_camino = torch.arange(N, dtype=torch.int32, device=dev)
addrs = torch.tensor([cs.data_ptr()], dtype=torch.int64, device=dev)
grupos = torch.zeros(1, dtype=torch.int32, device=dev)
bloques = sidx[None].contiguous()
hist = [[cs[int(sidx[n]), :, i].float().clone() for i in range(3)] for n in range(N)]   # ultimos 3 reales
acc = torch.ones(N, device=dev, dtype=torch.int32)


def arbol_al_azar(n):
    pad = torch.tensor([[int(torch.randint(-1, i, (1,), generator=gen)) for i in range(n)]])
    prof = torch.zeros_like(pad)
    for i in range(n):
        prof[0, i] = 0 if pad[0, i] < 0 else prof[0, pad[0, i]] + 1
    _, pad, _, _, _ = ab.orden_dfs(pad.clone(), pad, prof, pad.clone())
    return ab.padres_verificacion(pad)[0]


peor = 0.0
for paso in range(PASOS):
    x = torch.randn(N * T, DIM, device=dev, dtype=dt)
    padres = [torch.arange(-1, T - 1) if n == 0 else arbol_al_azar(SPEC) for n in range(N)]
    for n in range(N):
        pv = padres[n]
        for tt in range(T):      # ancestros 1..3 subiendo; pasado el ancla, historia -1/-2/-3
            c, j = [], (int(pv[tt]) if tt > 0 else -1)
            while len(c) < 3:
                c.append(j)
                j = int(pv[j]) if j > 0 else (j - 1 if j <= 0 else -1)
            anc3[n * T + tt] = torch.tensor(c, dtype=torch.int32, device=dev)
    ref = torch.zeros(N * T, DIM, device=dev)
    wf = w.float()
    for n in range(N):
        for t in range(T):
            cam, j = [], t
            while j >= 0:
                cam.append(x[n * T + j].float()); j = int(padres[n][j])
            sec = hist[n] + cam[::-1]
            z = sum(wf[:, i] * sec[-4 + i] for i in range(4))
            ref[n * T + t] = z * torch.sigmoid(z)
    cs_fus = cs_raw.clone().transpose(-1, -2)          # copia para el kernel FUSIONADO
    o_fus = ac.salidas(x, cs_fus, w, "silu", sidx, acc, cu, anc3, escribir_estado=True)
    o = ac.salidas(x, cs, w, "silu", sidx, acc, cu, anc3)
    x_up = x.clone()
    causal_conv1d_update(x_up, cs, w, None, "silu", conv_state_indices=sidx,
                         num_accepted_tokens=acc, query_start_loc=cu, max_query_len=T,
                         validate_data=False)
    # el estado que deja el kernel fusionado tiene que ser EL MISMO que el de upstream
    ok_est = all(torch.equal(cs_fus[int(sidx[n])], cs[int(sidx[n])]) for n in range(N))
    ok_sal = torch.equal(o_fus, o)
    err = ((o.float() - ref).norm(dim=1) / ref.norm(dim=1)).view(N, T).max(1).values
    peor = max(peor, float(err.max()))
    cadena_igual = torch.equal(o[:T], x_up[:T])
    # aceptar un camino al azar
    nuevo = []
    for n in range(N):
        hoja = int(torch.randint(0, T, (1,), generator=gen)); cam = []
        while hoja > 0:
            cam.append(hoja); hoja = int(padres[n][hoja])
        cam.reverse(); nuevo.append(len(cam) + 1)
        fila = torch.arange(SPEC, dtype=torch.int32); fila[:len(cam)] = torch.tensor([t - 1 for t in cam], dtype=torch.int32)
        camino[n] = fila.to(dev)
        hist[n] = (hist[n] + [x[n * T].float()] + [x[n * T + t].float() for t in cam])[-3:]
    acc = torch.tensor(nuevo, device=dev, dtype=torch.int32)
    antes = cs_raw.clone()
    if NEG != 1:
        ac.compactar(addrs, grupos, bloques, acc, camino, fila_camino, N,
                     (cs.stride(0), cs.stride(1), cs.stride(2)), DIM)
    torch.cuda.synchronize()
    intacto = torch.equal(antes[int(sidx[0])], cs_raw[int(sidx[0])])
    print(f"paso {paso:2d} err rel por pedido {[f'{e:.1e}' for e in err.tolist()]}  "
          f"fusionado==upstream: estado {ok_est} salidas {ok_sal}  cadena==upstream: {cadena_igual}  compactar no toco la cadena: {intacto}  acc->{nuevo}")
print(f"PEOR error relativo: {peor:.2e}")
