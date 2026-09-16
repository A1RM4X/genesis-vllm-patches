import os, sys, torch
sys.argv = ["x", "nada"]
src = open("/p/sk18g_paginado.py").read().rsplit("q, k, v = A.cargar(35); Nt", 1)[0]
exec(src)
q, k, v = A.cargar(35); kr = A.rotar_ptx(k)
for N in (22000, 22000, 20000):
    pos = torch.arange(N - 4, N, device=dev); qr = A.rotar_ptx(q[pos])
    lim = pos.repeat_interleave(G).to(torch.int32)
    p = preparar([qr[:, h * G:(h + 1) * G].reshape(-1, D) for h in range(2)], [kr[:N, h] for h in range(2)], [v[:N, h] for h in range(2)], lim, BS)
    p = paginar(p)
    lanzar_g(p); torch.cuda.synchronize()
    eh, el, em, es = emulador(p)
    bad = (p["oh"].long() != eh)
    badc = bad.any(-1).any(-1).nonzero().flatten().tolist()
    print(f"N={N}: tramos malos {badc}", flush=True)
    if badc:
        c = badc[0]
        br = bad[c].any(-1).nonzero().flatten().tolist(); bd = bad[c].any(0).nonzero().flatten().tolist()
        print("  filas", br[:20], "dims", bd[:20], "n dims", len(bd))
        r = br[0]; d = bd[0]
        print("  kernel", p["oh"][c, r, bd[:6]].tolist(), "\n  emul  ", eh[c, r, bd[:6]].tolist(), " bt", p["bt"][c].item())
