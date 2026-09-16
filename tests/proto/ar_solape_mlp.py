"""Solapamiento a nivel de BLOQUE MLP (TP=2, 2 procesos).

``ar_solape.py`` midio el solape del all-reduce con el GEMM que lo alimenta (``down_proj``) y dio
apenas 1,09x a M=8192: lo unico detras de lo que esconder la red es ese GEMM, y no alcanza.

La estructura correcta es partir el bloque MLP ENTERO en mitades de tokens. El MLP es por token
(gate_up, SiLU x gate, down son todos elemento a elemento en la dimension de token), asi que:

    mitad A: gate_up -> act -> down -> lanza el all-reduce de A en el stream lateral
    mitad B: gate_up -> act -> down          <- esto corre MIENTRAS viaja A
    espera A, lanza el all-reduce de B, espera B

Ahora el all-reduce de A se esconde detras de TODO el MLP de B (gate_up + act + down), que es
varias veces mas grande que un down_proj solo. Con N mitades el unico all-reduce que queda al
descubierto es el ultimo.

Ojo con NCCL_MAX_NCHANNELS: por omision NCCL toma muchos canales y sus kernels compiten por SM
con el GEMM. Medido con concurrencia pura (sin ninguna dependencia) el solape pasa de 23% a 42%
bajando los canales a 2.
"""
import os
import warnings

import torch

warnings.filterwarnings("ignore")
import torch.distributed as dist

H = 5120
I_RANK = 8704          # intermediate_size 17408 / TP 2
GRUPO = 64


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


def cuantizar(x, g=GRUPO):
    m, h = x.shape
    v = x.view(m, h // g, g)
    s = v.abs().amax(dim=-1).clamp_min(1e-6) / 127.0
    q = (v / s.unsqueeze(-1)).round().clamp_(-127, 127).to(torch.int8).view(m, h)
    return q, s


def sumar(qg, sg, m, w, dt):
    h = qg.shape[-1]
    ng = h // GRUPO
    out = qg[:m].view(m, ng, GRUPO).to(dt) * sg[:m].unsqueeze(-1).to(dt)
    for i in range(1, w):
        lo, hi = i * m, (i + 1) * m
        out = out + qg[lo:hi].view(m, ng, GRUPO).to(dt) * sg[lo:hi].unsqueeze(-1).to(dt)
    return out.view(m, h)


def act_quant(g):
    """SiLU(gate) * up y cuantizacion a int8 por token, como el camino W4A8 real."""
    a, b = g.chunk(2, dim=-1)
    y = torch.nn.functional.silu(a) * b
    s = y.abs().amax(-1, keepdim=True).clamp_min(1e-6) / 127.0
    return (y / s).round().clamp_(-127, 127).to(torch.int8), s.float()


_cuant_c = torch.compile(cuantizar, dynamic=False)
_sumar_c = torch.compile(sumar, dynamic=False)
_act_c = torch.compile(act_quant, dynamic=False)


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    w = dist.get_world_size()
    dev = f"cuda:{rank}"
    torch.manual_seed(rank)
    from vllm import _custom_ops as vops

    # PRIORIDAD ALTA: el MLP llena los 82 SM y el kernel de NCCL se queda esperando lugar. Con
    # prioridad alta sus bloques entran apenas se libera uno, que es lo que hace posible el solape.
    prio = int(os.environ.get("PRIO", "-1"))
    lo_p, hi_p = torch.cuda.Stream.priority_range() if hasattr(torch.cuda.Stream, "priority_range") \
        else (-1, 0)
    comm = torch.cuda.Stream(device=dev, priority=prio)

    def peso_i8(k, n):
        p_ = (torch.randn(k, n, device=dev, dtype=torch.float16) * 40).round()
        return p_.clamp(-127, 127).to(torch.int8).t().contiguous().t(), \
            torch.full((n,), 1e-4, device=dev, dtype=torch.float32)

    w_gu, s_gu = peso_i8(H, 2 * I_RANK)       # gate y up juntos
    w_dn, s_dn = peso_i8(I_RANK, H)

    def p(*a):
        if rank == 0:
            print(*a, flush=True)

    def mlp(xq, sx):
        """gate_up -> SiLU*gate -> quant -> down. Devuelve el parcial fp16 sin reducir."""
        g = vops.cutlass_scaled_mm(xq, w_gu, sx, s_gu, torch.float16)
        hq, sh = _act_c(g)
        return vops.cutlass_scaled_mm(hq, w_dn, sh, s_dn, torch.float16)

    def juntar(q, s):
        qg = torch.empty((q.shape[0] * w, q.shape[1]), dtype=torch.int8, device=q.device)
        sg = torch.empty((s.shape[0] * w, s.shape[1]), dtype=s.dtype, device=s.device)
        dist.all_gather_into_tensor(qg, q)
        dist.all_gather_into_tensor(sg, s)
        return qg, sg

    p(f"TP={w}  bloque MLP completo: [M,{H}]->[{2*I_RANK}] -> act -> [{I_RANK}]->[{H}] + all-reduce")
    p(f"canales NCCL = {os.environ.get('NCCL_MAX_NCHANNELS', 'por omision')}, "
      f"prioridad del stream = {prio} (rango {lo_p}..{hi_p})")
    p(f"{'M':>6} {'MLP':>7} {'AR':>7} {'serie':>8} {'pipe2':>8} {'pipe3':>8} {'pipe4':>8} "
      f"{'gana':>7} {'techo':>6}")

    for M in (8192, 4096, 2048):
        x = torch.randn(M, H, device=dev, dtype=torch.float16)
        xq = (x * 40).round().clamp(-127, 127).to(torch.int8)
        sx = torch.full((M, 1), 1e-3, device=dev, dtype=torch.float32)

        def solo_mlp():
            mlp(xq, sx)

        y0 = mlp(xq, sx)

        def solo_ar():
            q, s = _cuant_c(y0)
            qg, sg = juntar(q, s)
            _sumar_c(qg, sg, M, w, torch.float16)

        def serie():
            y = mlp(xq, sx)
            q, s = _cuant_c(y)
            qg, sg = juntar(q, s)
            return _sumar_c(qg, sg, M, w, torch.float16)

        def pipe(n):
            def f():
                paso = (M + n - 1) // n
                pend, outs = [], [None] * n
                for i in range(n):
                    lo, hi = i * paso, min(M, (i + 1) * paso)
                    if lo >= hi:
                        continue
                    y = mlp(xq[lo:hi], sx[lo:hi])            # main: todo el MLP de la mitad
                    q, sc = _cuant_c(y)                      # main
                    e = torch.cuda.Event(); e.record()
                    with torch.cuda.stream(comm):
                        comm.wait_event(e)
                        q.record_stream(comm); sc.record_stream(comm)
                        qg, sg = juntar(q, sc)               # lateral: solo la red
                        ec = torch.cuda.Event(); ec.record(comm)
                    pend.append((i, qg, sg, hi - lo, ec))
                for i, qg, sg, m_i, ec in pend:
                    torch.cuda.current_stream().wait_event(ec)
                    qg.record_stream(torch.cuda.current_stream())
                    sg.record_stream(torch.cuda.current_stream())
                    outs[i] = _sumar_c(qg, sg, m_i, w, torch.float16)
                return outs
            return f

        t_mlp = medir(solo_mlp)
        t_ar = medir(solo_ar)
        t_ser = medir(serie)
        ts = [medir(pipe(n)) for n in (2, 3, 4)]
        mejor = min(ts)
        # techo: el ultimo all-reduce nunca se esconde
        techo = (t_mlp + t_ar) / (t_mlp + t_ar / 2)
        p(f"{M:6d} {t_mlp:7.0f} {t_ar:7.0f} {t_ser:8.0f} {ts[0]:8.0f} {ts[1]:8.0f} {ts[2]:8.0f} "
          f"{t_ser/mejor:6.2f}x {techo:5.2f}x")

    # el pipeline tiene que dar exactamente lo mismo
    M = 2048
    x = torch.randn(M, H, device=dev, dtype=torch.float16)
    xq = (x * 40).round().clamp(-127, 127).to(torch.int8)
    sx = torch.full((M, 1), 1e-3, device=dev, dtype=torch.float32)
    q, s = _cuant_c(mlp(xq, sx))
    qg, sg = juntar(q, s)
    ref = _sumar_c(qg, sg, M, w, torch.float16)
    tr = []
    for i in range(4):
        lo, hi = i * (M // 4), (i + 1) * (M // 4)
        q, s = _cuant_c(mlp(xq[lo:hi], sx[lo:hi]))
        qg, sg = juntar(q, s)
        tr.append(_sumar_c(qg, sg, hi - lo, w, torch.float16))
    torch.cuda.synchronize()
    p(f"\nverificacion trozos vs entero: max dif = {(torch.cat(tr) - ref).abs().max().item():.2e}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
