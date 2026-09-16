"""PN136 offline: exactitud y tiempo de los buzones P2P reales, con 2 procesos.

Usa las clases de produccion (``vllm._genesis.p2p_buzon.Buzones``) contra un MLP de juguete con
las formas del modelo (H=5120, intermedio 8704 por rank), y compara:

    serie   = MLP entero + all-reduce de NCCL (lo de hoy)
    solapado= MLP partido en N trozos, cada trozo empujado por DMA mientras se calcula el siguiente

Comprueba que el resultado sea el mismo bit a bit (la cuantizacion int8 es identica en los dos
caminos, asi que la unica diferencia posible seria un error de indexado en los buzones).

OJO con la version anterior de este banco: compartia la memoria con reduce_tensor de torch, que
da 4,9 GB/s en vez de 13,0 y ademas deja la memoria fuera del alcance de los kernels. Con el IPC
hecho a mano (que es lo que hace ahora p2p_buzon) el transporte va a 13 GB/s y solapa 105%.
Ver tests/proto/p2p_origen_memoria.py.

Uso: torchrun --nproc_per_node=2 pn136_offline.py
"""
import math
import os
import warnings

import torch

warnings.filterwarnings("ignore")
import torch.distributed as dist

from vllm._genesis.p2p_buzon import Buzones

H = 5120
I_RANK = 8704
G = 64


def medir(f, n=10, calent=4):
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


def cuantizar(x, g=G):
    m, h = x.shape
    v = x.view(m, h // g, g)
    s = v.abs().amax(dim=-1).clamp_min(1e-6) / 127.0
    q = (v / s.unsqueeze(-1)).round().clamp_(-127, 127).to(torch.int8).view(m, h)
    return q, s.to(torch.float16)


def sumar(q_yo, s_yo, q_otros, s_otros, g, dt):
    m, h = q_yo.shape
    ng = h // g
    out = q_yo.view(m, ng, g).to(dt) * s_yo.unsqueeze(-1).to(dt)
    for i in range(q_otros.shape[0]):
        out = out + q_otros[i].view(m, ng, g).to(dt) * s_otros[i].unsqueeze(-1).to(dt)
    return out.view(m, h)


_c = torch.compile(cuantizar, dynamic=False)
_s = torch.compile(sumar, dynamic=False)


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    gcpu = dist.new_group(backend="gloo")
    w = dist.get_world_size()
    dev = f"cuda:{rank}"
    torch.manual_seed(rank)
    from vllm import _custom_ops as vops

    def p(*a):
        if rank == 0:
            print(*a, flush=True)

    def peso(k, n):
        t = (torch.randn(k, n, device=dev, dtype=torch.float16) * 40).round()
        return t.clamp(-127, 127).to(torch.int8).t().contiguous().t(), \
            torch.full((n,), 1e-4, device=dev, dtype=torch.float32)

    w_gu, s_gu = peso(H, 2 * I_RANK)
    w_dn, s_dn = peso(I_RANK, H)

    def act_q(g):
        a, b = g.chunk(2, dim=-1)
        y = torch.nn.functional.silu(a) * b
        sc = y.abs().amax(-1, keepdim=True).clamp_min(1e-6) / 127.0
        return (y / sc).round().clamp_(-127, 127).to(torch.int8), sc.float()

    act_c = torch.compile(act_q, dynamic=False)

    def parcial(xq, sx):
        g = vops.cutlass_scaled_mm(xq, w_gu, sx, s_gu, torch.float16)
        hq, sh = act_c(g)
        return vops.cutlass_scaled_mm(hq, w_dn, sh, s_dn, torch.float16)

    comm = torch.cuda.Stream(device=dev)
    devs = [0, 1]
    p(f"TP={w}  MLP [M,{H}]->[{2*I_RANK}]->act->[{I_RANK}]->[{H}]")
    p(f"{'M':>6} {'trozos':>7} {'serie':>9} {'solapado':>10} {'gana':>7} {'exacto':>7}"
      f"   | {'piezas':>40}")

    for M in (8192, 4096):
        for N in (1, 2, 4):
            cap = int(math.ceil(M / N))
            bz = Buzones(cap, H, G, N, rank, w, gcpu, devs)
            x = torch.randn(M, H, device=dev, dtype=torch.float16)
            xq = (x * 40).round().clamp(-127, 127).to(torch.int8)
            sx = torch.full((M, 1), 1e-3, device=dev, dtype=torch.float32)

            def serie():
                y = parcial(xq, sx)
                q, s = _c(y)
                qg = torch.empty(M * w, H, dtype=torch.int8, device=dev)
                sg = torch.empty(M * w, H // G, dtype=torch.float16, device=dev)
                dist.all_gather_into_tensor(qg, q)
                dist.all_gather_into_tensor(sg, s)
                otros = torch.stack([qg[i * M:(i + 1) * M] for i in range(w) if i != rank])
                otros_s = torch.stack([sg[i * M:(i + 1) * M] for i in range(w) if i != rank])
                return _s(q, s, otros, otros_s, G, torch.float16)

            def solapado():
                paso = int(math.ceil(M / N))
                pend = []
                for i in range(N):
                    lo, hi = i * paso, min(M, (i + 1) * paso)
                    if lo >= hi:
                        break
                    y = parcial(xq[lo:hi], sx[lo:hi])
                    q, s = _c(y)
                    ev = torch.cuda.Event(); ev.record()
                    with torch.cuda.stream(comm):
                        comm.wait_event(ev)
                        q.record_stream(comm); s.record_stream(comm)
                        e = bz.empujar(q, s, i, comm)
                    pend.append((i, lo, hi, q, s, e))
                out = torch.empty(M, H, dtype=torch.float16, device=dev)
                for i, lo, hi, q, s, e in pend:
                    bz.esperar(i, e, torch.cuda.current_stream())
                    dq, ds = bz.recibido(i, hi - lo)
                    out[lo:hi] = _s(q, s, dq, ds, G, torch.float16)
                return out

            ref = serie()
            got = solapado()
            torch.cuda.synchronize()
            exacto = bool((ref == got).all())

            # piezas, para ver donde se va el tiempo
            def solo_mlp():
                paso = int(math.ceil(M / N))
                for i in range(N):
                    lo, hi = i * paso, min(M, (i + 1) * paso)
                    if lo >= hi:
                        break
                    q, sc = _c(parcial(xq[lo:hi], sx[lo:hi]))

            def solo_cambio():
                """el intercambio solo, con la sincronizacion de verdad"""
                q = torch.zeros(M // N, H, dtype=torch.int8, device=dev)
                sc = torch.zeros(M // N, H // G, dtype=torch.float16, device=dev)
                pend = []
                for i in range(N):
                    e = bz.empujar(q, sc, i, torch.cuda.current_stream())
                    pend.append((i, e))
                for i, e in pend:
                    bz.esperar(i, e, torch.cuda.current_stream())

            t1 = medir(serie)
            t2 = medir(solapado)
            tm = medir(solo_mlp)
            tc = medir(solo_cambio)
            p(f"{M:6d} {N:7d} {t1:8.0f}u {t2:9.0f}u {t1/t2:6.2f}x {str(exacto):>7}"
              f"   | mlp {tm:7.0f}u  cambio DMA {tc:7.0f}u")
            del bz
            torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
