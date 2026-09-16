"""SK-18g: KV paginada real (bloques de 832 tokens en orden aleatorio, tabla de bloques).
1) exactitud bit a bit contra el emulador entero; 2) tiempo por capa contra FlashInfer fp8
con la MISMA paginacion (page_size 832, kv_indices desordenados) y con page_size 16."""
import os, sys, time, math, torch
os.environ.setdefault("SK18_VAR", "f")
MODO = sys.argv[1] if len(sys.argv) > 1 else "exacto"
sys.argv = [sys.argv[0]]
exec(open("/p/sk18e_test.py").read().split("q, k, v = A.cargar(35)")[0])
import flashinfer
BS = 832
kg = Kernel("sk18g_paged.cu", "sk18g_paged", defs=[f"-DQA={QA}", f"-DQB={QB}", "-DESPERA=" + os.environ.get("SK18_ESPERA", "0")], warps=4)
SHG = 32 * (256 + 128) + 2 * 64 * 256 + 2 * 256 * 64 + 2 * 64 * 2 * 4
torch.manual_seed(1)

def paginar(p, extra=8):
    NH, N = p["NH"], p["N"]
    NCH = (N + BS - 1) // BS; T = NCH * BS
    P = NCH + extra
    BLK = BS * NH * 256 * 2 + BS * NH * 4
    pool = torch.zeros(P, BLK, dtype=torch.int8, device=dev)
    bt = torch.randperm(P, device=dev)[:NCH].to(torch.int32)
    Kc = torch.zeros(NH, T, 256, dtype=torch.int8, device=dev); Kc[:, :N] = p["Kc"].view(NH, N, 256)
    Vd = torch.zeros(NH, 256, T, dtype=torch.int8, device=dev); Vd[:, :, :N] = p["Vt"].view(NH, 256, -1)[:, :, :N]
    E = torch.zeros(NH, T, 2, dtype=torch.int16, device=dev); E[:, :N, 0] = p["skf"].view(NH, N); E[:, :N, 1] = p["svf"].view(NH, N)
    KO = BS * NH * 256
    for c in range(NCH):
        b = pool[int(bt[c])]
        b[:KO] = Kc[:, c * BS:(c + 1) * BS].permute(1, 0, 2).reshape(-1)
        b[KO:2 * KO] = Vd[:, :, c * BS:(c + 1) * BS].reshape(-1)
        b[2 * KO:] = E[:, c * BS:(c + 1) * BS].permute(1, 0, 2).reshape(-1).view(torch.int8)
    p.update(pool=pool.contiguous(), bt=bt.contiguous(), CHK=BS, NCH=NCH, K=T,
             oh=torch.zeros(NCH, NH * p["MB"], D, dtype=torch.int32, device=dev), ol=torch.zeros(NCH, NH * p["MB"], D, dtype=torch.int32, device=dev),
             om=torch.zeros(NCH, NH * p["MB"], dtype=torch.int32, device=dev), os=torch.zeros(NCH, NH * p["MB"], dtype=torch.int32, device=dev))
    return p

def lanzar_g(p):
    kg.lanzar((p["NCH"], p["NH"] * p["MB"] // 32), [p["Q"], p["pool"], p["bt"], p["lim"], p["mqb"], p["dcap"],
                                                   p["oh"], p["ol"], p["om"], p["os"], p["MB"], p["N"], p["K"], p["CHK"], p["NCH"], p["NH"]], shared=SHG)

def medir(f, n=30):
    for _ in range(5): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e3

q, k, v = A.cargar(35); Nt = k.shape[0]; kr = A.rotar_ptx(k)
modo = MODO
if modo == "exacto":
    for N in (700, 3000, 22000, 22000, 20000):
        pos = torch.arange(N - 4, N, device=dev); qr = A.rotar_ptx(q[pos])
        lim = pos.repeat_interleave(G).to(torch.int32)
        p = preparar([qr[:, h * G:(h + 1) * G].reshape(-1, D) for h in range(2)], [kr[:N, h] for h in range(2)], [v[:N, h] for h in range(2)], lim, BS)
        p = paginar(p)
        lanzar_g(p); torch.cuda.synchronize()
        eh, el, em, es = emulador(p)
        ok = [bool((p["oh"].long() == eh).all()), bool((p["ol"].long() == el).all()), bool((p["om"].long() == em).all()), bool((p["os"].long() == es).all())]
        Ot, St = unir_ptx(p)
        ref = A.referencia(q, k, v, pos)
        out = torch.stack([Ot[h * p["MB"]:h * p["MB"] + 24].double() * p["svmax"][h] / St[h * p["MB"]:h * p["MB"] + 24].double().clamp_min(1)[:, None] for h in range(2)])
        r = torch.stack([ref[:, h * G:(h + 1) * G].reshape(-1, D) for h in range(2)]).double()
        print(f"N={N} paginas={p['NCH']}: hi/lo/m/S iguales = {ok}  error vs float {100*((out - r).norm(dim=-1) / r.norm(dim=-1)).mean().item():.3f}%", flush=True)
else:
    qr = A.rotar_ptx(q[Nt - 4:])
    lim0 = torch.zeros(24, dtype=torch.int32, device=dev)
    for N in (16000, 57000, 100000):
        reps = (N + Nt - 1) // Nt
        lim = (N - 4 + torch.arange(4, device=dev)).repeat_interleave(G).to(torch.int32)
        p = preparar([qr[:, h * G:(h + 1) * G].reshape(-1, D) for h in range(2)], [kr[:, h].repeat(reps, 1)[:N] for h in range(2)],
                     [v[:, h].repeat(reps, 1)[:N] for h in range(2)], lim, BS)
        p = paginar(p)
        del p["Kc"], p["Vt"], p["Vpag"]; torch.cuda.empty_cache()
        tg = medir(lambda: (lanzar_g(p), unir_ptx(p)))
        tk = medir(lambda: lanzar_g(p))
        # FlashInfer fp8, 12 q / 2 kv, misma paginacion desordenada
        res = []
        for PAGE in (BS, 16):
            npag = (N + PAGE - 1) // PAGE
            Ptot = npag + 8
            kv = (torch.randn(Ptot, 2, PAGE, 2, D, device=dev) * 0.5).to(torch.float8_e4m3fn)
            ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
            w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
            qo = torch.tensor([0, 4], dtype=torch.int32, device=dev); kvi = torch.tensor([0, npag], dtype=torch.int32, device=dev)
            idx = torch.randperm(Ptot, device=dev)[:npag].to(torch.int32)
            last = torch.tensor([N - (npag - 1) * PAGE], dtype=torch.int32, device=dev)
            w.plan(qo, kvi, idx, last, 12, 2, D, PAGE, causal=True, pos_encoding_mode="NONE", q_data_type=torch.float16, kv_data_type=torch.float8_e4m3fn)
            qq = torch.randn(4, 12, D, device=dev, dtype=torch.float16)
            res.append(medir(lambda: w.run(qq, kv)))
            del kv, ws, w; torch.cuda.empty_cache()
        print(f"N={N} ({p['NCH']} paginas): SK-18g kernel {tk:.3f} + union = {tg:.3f} ms/capa | FlashInfer fp8 pag 832 {res[0]:.3f} | pag 16 {res[1]:.3f}", flush=True)
