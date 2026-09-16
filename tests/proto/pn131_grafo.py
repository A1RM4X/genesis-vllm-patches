"""Costo del decode de PN131 en la forma del CUDA graph (NCH = maximo) contra eager (NCH justo),
por componente, a 16k/57k/100k (q/k/v reales de la capa 35 repetidas)."""
import sys, types, time, torch
exec(open("/p/pn131_offline.py").read().split("q, k, v = A.cargar(35)")[0])
q, k, v = A.cargar(35); Nt = k.shape[0]
def medir(f, n=30):
    for _ in range(5): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e3
for N in (16000, 57000, 100000):
    reps = (N + Nt - 1) // Nt
    kk = k.repeat(reps, 1, 1)[:N]; vv = v.repeat(reps, 1, 1)[:N]
    nblocks = (N + BS - 1) // BS + 4
    kv = kv_nuevo(nblocks)
    perm = torch.randperm(nblocks)[: (N + BS - 1) // BS].to(torch.int32)
    slots = (perm[torch.arange(N) // BS].to(torch.int64) * BS + torch.arange(N) % BS).to(dev)
    for t0 in range(0, N, 7488):
        P.escribir(impl, layer, kk[t0:t0 + 7488].half(), vv[t0:t0 + 7488].half(), kv, slots[t0:t0 + 7488])
    bt = torch.zeros(1, 316, dtype=torch.int32); bt[0, :perm.numel()] = perm
    md = meta(torch.tensor([0, 4], dtype=torch.int32), torch.tensor([N], dtype=torch.int32), bt, 4)
    qq = q[Nt - 4:].half()
    out = torch.zeros(4, 12, D, dtype=torch.float16, device=dev)
    te = medir(lambda: P._decode_uniforme(impl, qq, kv, md, out, 1, 4, False))
    tg = medir(lambda: P._decode_uniforme(impl, qq, kv, md, out, 1, 4, True))
    tw = medir(lambda: P.escribir(impl, layer, kk[N-4:N].half(), vv[N-4:N].half(), kv, slots[N-4:N]))
    print(f"N={N}: decode eager {te:.3f} ms | forma grafo (NCH={P._bufs[0].nchmax}) {tg:.3f} ms | escritura {tw:.3f} ms", flush=True)
