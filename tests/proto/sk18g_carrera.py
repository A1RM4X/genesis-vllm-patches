import os, sys, torch
sys.argv = ["x"]
src = open("/p/sk18g_paginado.py").read().rsplit("q, k, v = A.cargar(35); Nt", 1)[0]
exec(src)
q, k, v = A.cargar(35); kr = A.rotar_ptx(k)
N = 22000
pos = torch.arange(N - 4, N, device=dev); qr = A.rotar_ptx(q[pos])
lim = pos.repeat_interleave(G).to(torch.int32)
p = preparar([qr[:, h * G:(h + 1) * G].reshape(-1, D) for h in range(2)], [kr[:N, h] for h in range(2)], [v[:N, h] for h in range(2)], lim, BS)
eh = None
for esp in (1, 0):
    kgx = Kernel("sk18g_paged.cu", "sk18g_paged", defs=[f"-DQA={QA}", f"-DQB={QB}", f"-DESPERA={esp}"], warps=4)
    kfx = Kernel("sk18f_stream.cu", "sk18f_stream", defs=[f"-DQA={QA}", f"-DQB={QB}", f"-DESPERA={esp}", "-DNWARPS=4"], warps=4)
    for rep in range(3):
        pg = paginar(dict(p))
        if eh is None: eh, el, em, es = emulador(pg)
        kgx.lanzar((pg["NCH"], 2 * pg["MB"] // 32), [pg["Q"], pg["pool"], pg["bt"], pg["lim"], pg["mqb"], pg["dcap"], pg["oh"], pg["ol"], pg["om"], pg["os"], pg["MB"], pg["N"], pg["K"], pg["CHK"], pg["NCH"], pg["NH"]], shared=SHG)
        torch.cuda.synchronize()
        okg = bool((pg["oh"].long() == eh).all() and (pg["ol"].long() == el).all() and (pg["om"].long() == em).all())
        pf = dict(p); pf.update(CHK=BS, NCH=pg["NCH"], oh=torch.zeros_like(pg["oh"]), ol=torch.zeros_like(pg["ol"]), om=torch.zeros_like(pg["om"]), os=torch.zeros_like(pg["os"]))
        pf["esc"] = escalas(pf)
        kfx.lanzar((pf["NCH"], 2 * pf["MB"] // 32), [pf["Q"], pf["Kc"], pf["Vpag"], pf["esc"], pf["lim"], pf["mqb"], pf["dcap"], pf["oh"], pf["ol"], pf["om"], pf["os"], pf["MB"], pf["N"], pf["K"], pf["CHK"], pf["NCH"], pf["NH"]], shared=SH)
        torch.cuda.synchronize()
        okf = bool((pf["oh"].long() == eh).all() and (pf["ol"].long() == el).all() and (pf["om"].long() == em).all())
        print(f"ESPERA={esp} rep {rep}: SK-18g exacto={okg}  SK-18f(CHK 832) exacto={okf}", flush=True)
