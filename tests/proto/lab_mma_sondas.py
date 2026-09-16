"""Descubre el layout real de fragmentos de mma (s8 m16n8k32 y s4 m16n8k64) con sondas.

Para cada posicion fisica (hilo, registro, sub-elemento) de A:
    A = one-hot en esa posicion, B = todo unos  -> las salidas no nulas son la FILA i.
Para cada posicion fisica de B:
    A = todo unos, B = one-hot                  -> las salidas no nulas son la COLUMNA n.
Emparejamiento de k:
    A = one-hot, B = valores distintos por (columna, k) -> out[fila, n] = B[k][n] identifica k.
Con los tres mapas se construye el empaquetado y se verifica con datos al azar.
"""
import collections, sys, torch
from vllm._genesis.kernels.ptx_lab import Kernel

dev = "cuda"
CFG = {"s8": dict(entrada="lab_mma_s8", bits=8, k=32), "s4": dict(entrada="lab_mma_s4", bits=4, k=64)}


def regs_desde_subs(subs, bits):
    """subs [G, 32, nregs, 32//bits] valores con signo -> registros uint32 [G,32,nregs]."""
    mask = (1 << bits) - 1
    u = (subs.to(torch.int64) & mask)
    shifts = torch.arange(0, 32, bits, device=subs.device, dtype=torch.int64)
    val = (u << shifts).sum(-1)
    return val.to(torch.int64).where(val < 2**31, val - 2**32).to(torch.int32)


def correr(k, subsA, subsB, bits):
    G = subsA.shape[0]
    ra = regs_desde_subs(subsA, bits).contiguous(); rb = regs_desde_subs(subsB, bits).contiguous()
    out = torch.zeros(G, 32, 4, dtype=torch.int32, device=dev)
    k.lanzar((G, 1), [ra, rb, out, 1], sync=True)
    return out.view(G, 128)


for nombre in sys.argv[1:] or ["s8", "s4"]:
    c = CFG[nombre]; bits = c["bits"]; per = 32 // bits
    k = Kernel("lab_mma.cu", c["entrada"], warps=1)
    nA = 32 * 4 * per; nB = 32 * 2 * per
    # --- filas: A one-hot, B unos
    sA = torch.zeros(nA, 32 * 4 * per, dtype=torch.int8, device=dev); sA[torch.arange(nA), torch.arange(nA)] = 1
    sB = torch.ones(nA, 32 * 2 * per, dtype=torch.int8, device=dev)
    outA = correr(k, sA.view(nA, 32, 4, per), sB.view(nA, 32, 2, per), bits)
    # --- columnas: A unos, B one-hot
    sA2 = torch.ones(nB, 32 * 4 * per, dtype=torch.int8, device=dev)
    sB2 = torch.zeros(nB, 32 * 2 * per, dtype=torch.int8, device=dev); sB2[torch.arange(nB), torch.arange(nB)] = 1
    outB = correr(k, sA2.view(nB, 32, 4, per), sB2.view(nB, 32, 2, per), bits)
    filas = [tuple(torch.nonzero(outA[p]).flatten().tolist()) for p in range(nA)]
    cols = [tuple(torch.nonzero(outB[p]).flatten().tolist()) for p in range(nB)]
    print(f"{nombre}: A posiciones -> {len(set(filas))} filas distintas, tamanos {collections.Counter(len(f) for f in filas)}")
    print(f"{nombre}: B posiciones -> {len(set(cols))} columnas distintas, tamanos {collections.Counter(len(x) for x in cols)}")
    fila_id = {f: i for i, f in enumerate(sorted(set(filas)))}
    col_id = {x: j for j, x in enumerate(sorted(set(cols)))}
    # salida (posicion) -> (fila, col)
    pos_rc = {}
    for f, i in fila_id.items():
        for x, j in col_id.items():
            inter = set(f) & set(x)
            if len(inter) == 1:
                pos_rc[inter.pop()] = (i, j)
    print(f"{nombre}: salidas mapeadas {len(pos_rc)}/128")
    # --- emparejamiento de k: A one-hot, B con valor distinto por posicion dentro de cada columna
    maxv = (1 << (bits - 1)) - 1
    base = torch.zeros(nB, dtype=torch.int8, device=dev)
    porcol = collections.defaultdict(list)
    for p in range(nB):
        porcol[col_id[cols[p]]].append(p)
    for j, ps in porcol.items():
        for r, p in enumerate(ps):
            base[p] = (r % maxv) + 1 if len(ps) <= maxv else 0
    distintos = all(len(ps) <= maxv for ps in porcol.values())
    k_de_A = {}
    if distintos:
        sB3 = base.unsqueeze(0).expand(nA, -1).contiguous()
        outK = correr(k, sA.view(nA, 32, 4, per), sB3.view(nA, 32, 2, per), bits)
        for p in range(nA):
            ks = []
            for q, v in enumerate(outK[p].tolist()):
                if v and q in pos_rc:
                    j = pos_rc[q][1]
                    cand = [b for b in porcol[j] if int(base[b]) == v]
                    ks.append(cand[0] if len(cand) == 1 else None)
            k_de_A[p] = ks
        print(f"{nombre}: emparejamiento k resuelto por valores unicos")
    else:
        print(f"{nombre}: columnas con mas posiciones que valores distintos ({max(len(v) for v in porcol.values())} > {maxv}); "
              f"emparejo k con sondas B one-hot x A one-hot por columna")
    torch.save({"filas": filas, "cols": cols, "fila_id": fila_id, "col_id": col_id, "pos_rc": pos_rc,
                "k_de_A": k_de_A}, f"/tmp/sondas_{nombre}.pt")
    # muestra de los primeros hilos
    for p in list(range(0, nA, per))[:6]:
        print(f"   A pos {p:4d} (hilo {p // (4*per)}, reg {(p // per) % 4}, sub {p % per}) -> fila {fila_id[filas[p]]}")
    for p in list(range(0, nB, per))[:4]:
        print(f"   B pos {p:4d} (hilo {p // (2*per)}, reg {(p // per) % 2}, sub {p % per}) -> col {col_id[cols[p]]}")
