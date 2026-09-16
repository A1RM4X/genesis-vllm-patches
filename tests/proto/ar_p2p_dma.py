"""Intercambio P2P por el motor DMA en vez de NCCL (TP=2, 2 procesos).

Por que
-------
Con TP=2 un all-reduce es UN intercambio entre pares mas una suma local. NCCL lo hace con kernels
(protocolo LL: los SM copian a mano, porque para mensajes chicos es la menor latencia) y por eso
COMPITE con el GEMM por SM. Medido en ``ar_solape.py``:

  * concurrencia pura (sin ninguna dependencia): NCCL solapa 23% con el GEMM, y 42% bajando
    NCCL_MAX_NCHANNELS a 2;
  * a nivel de bloque MLP el solape se va a CERO, porque el MLP llena los 82 SM y los bloques de
    NCCL no consiguen lugar (darle prioridad alta al stream tampoco alcanza).

Una ``cudaMemcpyPeerAsync`` en cambio la ejecuta el MOTOR DMA de la placa, que no usa SM ninguno:
solapa con el computo por construccion. Las dos 3090 estan en el mismo puente PCIe (PIX) y tienen
P2P habilitado, asi que la copia va directa de una memoria a la otra sin pasar por la RAM del
host.

Este banco mide las dos cosas contra el mismo trabajo de computo.
"""
import os
import pickle
import warnings

import torch

warnings.filterwarnings("ignore")
import torch.distributed as dist
import torch.multiprocessing.reductions as mpr

H = 5120
I_RANK = 8704


def medir(f, n=20, calent=5):
    for _ in range(calent):
        f()
    torch.cuda.synchronize()
    dist.barrier()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(n):
        f()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n * 1000


def compartir(t, grupo_cpu, rank, w):
    """Publica el tensor por IPC y devuelve la lista de vistas de todos los ranks."""
    datos = pickle.dumps(mpr.reduce_tensor(t))
    buf = torch.frombuffer(bytearray(datos), dtype=torch.uint8)
    n = torch.tensor([buf.numel()], dtype=torch.int64)
    todos_n = [torch.zeros(1, dtype=torch.int64) for _ in range(w)]
    dist.all_gather(todos_n, n, group=grupo_cpu)
    mx = int(max(int(x.item()) for x in todos_n))
    pad = torch.zeros(mx, dtype=torch.uint8)
    pad[:buf.numel()] = buf
    recib = [torch.zeros(mx, dtype=torch.uint8) for _ in range(w)]
    dist.all_gather(recib, pad, group=grupo_cpu)
    fuera = []
    for i in range(w):
        if i == rank:
            fuera.append(t)
        else:
            f, a = pickle.loads(bytes(recib[i][:int(todos_n[i].item())].numpy()))
            fuera.append(f(*a))
    return fuera


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    gcpu = dist.new_group(backend="gloo")
    w = dist.get_world_size()
    dev = f"cuda:{rank}"
    torch.manual_seed(rank)
    from vllm import _custom_ops as vops

    comm = torch.cuda.Stream(device=dev)

    def p(*a):
        if rank == 0:
            print(*a, flush=True)

    pw = (torch.randn(I_RANK, H, device=dev, dtype=torch.float16) * 40).round()
    pw = pw.clamp(-127, 127).to(torch.int8).t().contiguous().t()
    sw = torch.full((H,), 1e-4, device=dev, dtype=torch.float32)

    p(f"TP={w}  intercambio de [M, {H}] int8 entre las dos placas")
    p(f"{'M':>6} {'MB':>6} {'GEMM':>7} {'NCCL':>7} {'DMA':>7} | "
      f"{'GEMM+NCCL':>10} {'GEMM+DMA':>9} | {'solape NCCL':>12} {'solape DMA':>11}")

    for M in (8192, 4096, 2048):
        x = torch.randn(M, I_RANK, device=dev, dtype=torch.float16)
        xq = (x * 40).round().clamp(-127, 127).to(torch.int8)
        sx = torch.full((M, 1), 1e-3, device=dev, dtype=torch.float32)

        # buffer propio (lo que este rank publica) y buffers de todos, mapeados por IPC
        mio = torch.zeros(M, H, dtype=torch.int8, device=dev)
        vistas = compartir(mio, gcpu, rank, w)
        buzon = torch.zeros(M, H, dtype=torch.int8, device=dev)   # donde recibe este rank
        buzones = compartir(buzon, gcpu, rank, w)
        qg = torch.empty(M * w, H, dtype=torch.int8, device=dev)

        def gemm():
            vops.cutlass_scaled_mm(xq, pw, sx, sw, torch.float16)

        def nccl():
            dist.all_gather_into_tensor(qg, mio)

        def dma():
            """Cada rank ESCRIBE su parcial en el buzon del otro: una sola copia P2P por rank."""
            for i in range(w):
                if i != rank:
                    buzones[i].copy_(mio, non_blocking=True)

        def juntos(red):
            def f():
                e = torch.cuda.Event(); e.record()
                with torch.cuda.stream(comm):
                    comm.wait_event(e)
                    red()
                    ec = torch.cuda.Event(); ec.record(comm)
                gemm()
                torch.cuda.current_stream().wait_event(ec)
            return f

        t_g = medir(gemm)
        t_n = medir(nccl)
        t_d = medir(dma)
        t_gn = medir(juntos(nccl), n=10)
        t_gd = medir(juntos(dma), n=10)
        mb = M * H / 1e6
        sn = 100 * (t_g + t_n - t_gn) / min(t_g, t_n)
        sd = 100 * (t_g + t_d - t_gd) / min(t_g, t_d)
        p(f"{M:6d} {mb:6.1f} {t_g:7.0f} {t_n:7.0f} {t_d:7.0f} | "
          f"{t_gn:10.0f} {t_gd:9.0f} | {sn:11.0f}% {sd:10.0f}%")

    # ── el experimento que decide: bloque MLP completo, partido, con los dos transportes ──
    p("\nbloque MLP completo partido en trozos (el all-reduce del trozo i viaja mientras se")
    p("calcula el trozo i+1). serie = sin partir, con NCCL.")
    w_gu = (torch.randn(H, 2 * I_RANK, device=dev, dtype=torch.float16) * 40).round()
    w_gu = w_gu.clamp(-127, 127).to(torch.int8).t().contiguous().t()
    s_gu = torch.full((2 * I_RANK,), 1e-4, device=dev, dtype=torch.float32)
    w_dn = (torch.randn(I_RANK, H, device=dev, dtype=torch.float16) * 40).round()
    w_dn = w_dn.clamp(-127, 127).to(torch.int8).t().contiguous().t()
    s_dn = torch.full((H,), 1e-4, device=dev, dtype=torch.float32)

    def act_q(g):
        a, b = g.chunk(2, dim=-1)
        y = torch.nn.functional.silu(a) * b
        sc = y.abs().amax(-1, keepdim=True).clamp_min(1e-6) / 127.0
        return (y / sc).round().clamp_(-127, 127).to(torch.int8), sc.float()

    act_c = torch.compile(act_q, dynamic=False)

    def bloque_mlp(xi, si):
        g = vops.cutlass_scaled_mm(xi, w_gu, si, s_gu, torch.float16)
        hq, sh = act_c(g)
        return vops.cutlass_scaled_mm(hq, w_dn, sh, s_dn, torch.float16)

    def q_tok(y):
        sc = y.abs().amax(-1, keepdim=True).clamp_min(1e-6) / 127.0
        return (y / sc).round().clamp_(-127, 127).to(torch.int8), sc.float()

    qtok_c = torch.compile(q_tok, dynamic=False)

    p(f"{'M':>6} {'MLP':>7} {'serieN':>8} {'pipeN2':>8} {'pipeN4':>8} {'pipeD2':>8} "
      f"{'pipeD4':>8} {'pipeD8':>8} {'gana':>7}")
    for M in (8192, 4096):
        x = torch.randn(M, H, device=dev, dtype=torch.float16)
        xq = (x * 40).round().clamp(-127, 127).to(torch.int8)
        sx = torch.full((M, 1), 1e-3, device=dev, dtype=torch.float32)
        # buzones por trozo, reservados una vez (en un grafo real serian persistentes)
        bz = {}
        for n in (1, 2, 4, 8):
            paso = (M + n - 1) // n
            mios = [torch.zeros(paso, H, dtype=torch.int8, device=dev) for _ in range(n)]
            recs = [torch.zeros(paso * w, H, dtype=torch.int8, device=dev) for _ in range(n)]
            bz[n] = (mios, [compartir(r, gcpu, rank, w) for r in recs], recs)

        def serie_n():
            y = bloque_mlp(xq, sx)
            q, sc = qtok_c(y)
            g_ = torch.empty(M * w, H, dtype=torch.int8, device=dev)
            dist.all_gather_into_tensor(g_, q)
            return g_

        def pipe(n, dma):
            mios, vis, recs = bz[n]
            def f():
                paso = (M + n - 1) // n
                pend = []
                for i in range(n):
                    lo, hi = i * paso, min(M, (i + 1) * paso)
                    if lo >= hi:
                        continue
                    y = bloque_mlp(xq[lo:hi], sx[lo:hi])
                    q, sc = qtok_c(y)
                    mios[i][:hi - lo].copy_(q)
                    e = torch.cuda.Event(); e.record()
                    with torch.cuda.stream(comm):
                        comm.wait_event(e)
                        mios[i].record_stream(comm)
                        if dma:
                            for r in range(w):
                                vis[i][r][rank * paso:rank * paso + (hi - lo)].copy_(
                                    mios[i][:hi - lo], non_blocking=True)
                        else:
                            dist.all_gather_into_tensor(recs[i], mios[i])
                        ec = torch.cuda.Event(); ec.record(comm)
                    pend.append(ec)
                for ec in pend:
                    torch.cuda.current_stream().wait_event(ec)
            return f

        t_m = medir(lambda: bloque_mlp(xq, sx), n=8)
        t_s = medir(serie_n, n=8)
        tn2, tn4 = medir(pipe(2, False), n=8), medir(pipe(4, False), n=8)
        td2, td4, td8 = (medir(pipe(2, True), n=8), medir(pipe(4, True), n=8),
                         medir(pipe(8, True), n=8))
        mejor = min(tn2, tn4, td2, td4, td8)
        p(f"{M:6d} {t_m:7.0f} {t_s:8.0f} {tn2:8.0f} {tn4:8.0f} {td2:8.0f} {td4:8.0f} "
          f"{td8:8.0f} {t_s/mejor:6.2f}x")

    # correccion: el buzon tiene que traer de verdad lo del otro rank
    M = 1024
    mio = torch.full((M, H), rank + 1, dtype=torch.int8, device=dev)
    buzon = torch.zeros(M, H, dtype=torch.int8, device=dev)
    buzones = compartir(buzon, gcpu, rank, w)
    for i in range(w):
        if i != rank:
            buzones[i].copy_(mio)
    torch.cuda.synchronize()
    dist.barrier()
    esperado = (1 - rank) + 1       # con w=2, lo que escribio el otro
    ok = bool((buzon == esperado).all())
    p(f"\nverificacion del intercambio P2P: buzon trae lo del otro rank = {ok}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
