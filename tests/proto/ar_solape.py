"""Solapamiento del all-reduce con el GEMM que lo alimenta, en el prefill (TP=2, 2 procesos).

En prefill el all-reduce es el 41,6% del tiempo de GPU y ya corre a 10,8 GB/s, o sea al techo del
PCIe: comprimir mas no se puede (el int4 esta dominado, ver ar_int4_numerica.py). Lo que queda es
que la comunicacion pase MIENTRAS se calcula.

La idea: ``down_proj`` y ``o_proj`` son por token, asi que el bloque se puede partir en trozos de
filas. Se calcula el trozo i y se manda por la red en un stream aparte mientras el SM sigue con el
trozo i+1. El techo del ahorro es el menor de los dos tiempos.

Ojo con el antecedente: el async TP de vLLM ya se probo y dio 3-4% PEOR. Este banco mide el caso
a mano, sin sequence parallelism, para ver si el problema era la implementacion o el techo.

Uso:  torchrun --nproc_per_node=2 ar_solape.py     (o el lanzador de abajo, que hace el spawn)
"""
import os
import warnings
import torch

warnings.filterwarnings("ignore")
import torch.distributed as dist

H = 5120
I_RANK = 8704          # intermediate_size 17408 / TP 2
GRUPO = 64             # el de PN120


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
    return a.elapsed_time(b) / n * 1000      # us


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


_cuant_c = torch.compile(cuantizar, dynamic=False)
_sumar_c = torch.compile(sumar, dynamic=False)


def ar_int8(y, w):
    q, s = _cuant_c(y)
    qg, sg = _juntar(q, s, w)
    return _sumar_c(qg, sg, y.shape[0], w, y.dtype)


def _juntar(q, s, w):
    """Solo la parte que habla por la red."""
    qg = torch.empty((q.shape[0] * w, q.shape[1]), dtype=torch.int8, device=q.device)
    sg = torch.empty((s.shape[0] * w, s.shape[1]), dtype=s.dtype, device=s.device)
    dist.all_gather_into_tensor(qg, q)
    dist.all_gather_into_tensor(sg, s)
    return qg, sg


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    w = dist.get_world_size()
    dev = f"cuda:{rank}"
    torch.manual_seed(rank)

    comm = torch.cuda.Stream(device=dev)
    peso = torch.randn(I_RANK, H, device=dev, dtype=torch.float16) * 0.02
    # sustituto honesto de Marlin W4A8: el GEMM int8 de cutlass corre a la misma tasa de tensor
    # core que el de Marlin (2x el fp16), asi que da la relacion GEMM/comunicacion realista.
    from vllm import _custom_ops as vops
    peso_q = (peso.t().contiguous() * 60).round().clamp(-127, 127).to(torch.int8).t().contiguous()
    esc_w = torch.full((H,), 1 / 60.0, device=dev, dtype=torch.float32)

    def p(*a):
        if rank == 0:
            print(*a, flush=True)

    p(f"TP={w}  down_proj [M, {I_RANK}] @ [{I_RANK}, {H}]  +  all-reduce de [M, {H}]")
    p(f"{'M':>6} {'GEMMf16':>8} {'GEMMi8':>8} {'ARfp16':>8} {'ARint8':>8} | "
      f"{'serie':>8} {'todo4':>8} {'red2':>8} {'red4':>8} {'red8':>8} {'gana':>7} {'techo':>6}")

    for M in (8192, 4096, 2048):
        x = torch.randn(M, I_RANK, device=dev, dtype=torch.float16)
        y0 = torch.empty(M, H, device=dev, dtype=torch.float16)

        xq = (x * 40).round().clamp(-127, 127).to(torch.int8)
        esc_x = torch.full((M, 1), 1 / 40.0, device=dev, dtype=torch.float32)
        pq = peso_q.t().contiguous().t()          # [K, N] en column-major, como pide cutlass

        def gemm_i8(xi, si):
            return vops.cutlass_scaled_mm(xi, pq, si, esc_w, torch.float16)

        def solo_gemm():
            torch.mm(x, peso, out=y0)

        def solo_gemm_i8():
            gemm_i8(xq, esc_x)

        def solo_ar_fp16():
            dist.all_reduce(y0)

        def solo_ar_int8():
            ar_int8(y0, w)

        t_gemm = medir(solo_gemm)
        try:
            t_g8 = medir(solo_gemm_i8)
        except Exception as e:
            p(f"  (GEMM int8 no disponible: {e})")
            t_g8 = float("nan")
        t_arf = medir(solo_ar_fp16)
        t_ari = medir(solo_ar_int8)

        def serie():
            y = gemm_i8(xq, esc_x)
            return ar_int8(y, w)

        def pipe_nccl(n):
            """Solo el NCCL va al stream lateral. La cuantizacion y la suma son kernels de
            COMPUTO: si se los manda al stream de comunicacion compiten por SM con el GEMM y el
            solape se evapora (medido: 1,06x contra 1,9x de techo)."""
            def f():
                paso = (M + n - 1) // n
                pend, outs = [], [None] * n
                for i in range(n):
                    lo, hi = i * paso, min(M, (i + 1) * paso)
                    if lo >= hi:
                        continue
                    y = gemm_i8(xq[lo:hi], esc_x[lo:hi])       # main
                    q, sc = _cuant_c(y)                        # main
                    e = torch.cuda.Event(); e.record()
                    with torch.cuda.stream(comm):
                        comm.wait_event(e)
                        q.record_stream(comm); sc.record_stream(comm)
                        qg, sg = _juntar(q, sc, w)             # lateral: solo la red
                        ec = torch.cuda.Event(); ec.record(comm)
                    pend.append((i, qg, sg, hi - lo, ec))
                for i, qg, sg, m_i, ec in pend:
                    torch.cuda.current_stream().wait_event(ec)
                    qg.record_stream(torch.cuda.current_stream())
                    sg.record_stream(torch.cuda.current_stream())
                    outs[i] = _sumar_c(qg, sg, m_i, w, torch.float16)   # main
                return outs
            return f

        def pipe(n):
            def f():
                paso = (M + n - 1) // n
                ys, evg, evc, outs = [], [], [], [None] * n
                for i in range(n):
                    lo, hi = i * paso, min(M, (i + 1) * paso)
                    if lo >= hi:
                        continue
                    y = gemm_i8(xq[lo:hi], esc_x[lo:hi])
                    e = torch.cuda.Event()
                    e.record()
                    ys.append(y)
                    evg.append(e)
                    with torch.cuda.stream(comm):
                        comm.wait_event(e)
                        y.record_stream(comm)
                        outs[i] = ar_int8(y, w)
                        outs[i].record_stream(torch.cuda.current_stream())
                        ec = torch.cuda.Event()
                        ec.record(comm)
                        evc.append(ec)
                for e in evc:
                    torch.cuda.current_stream().wait_event(e)
                return outs
            return f

        # diagnostico: GEMM y NCCL a la vez SIN dependencia ninguna. Si esto no baja de la suma,
        # el solape es imposible en este hardware y no hay implementacion que lo arregle.
        qq = torch.empty(M, H, dtype=torch.int8, device=dev)
        qqg = torch.empty(M * w, H, dtype=torch.int8, device=dev)

        def concurrente():
            e = torch.cuda.Event(); e.record()
            with torch.cuda.stream(comm):
                comm.wait_event(e)
                dist.all_gather_into_tensor(qqg, qq)
                ec = torch.cuda.Event(); ec.record(comm)
            gemm_i8(xq, esc_x)
            torch.cuda.current_stream().wait_event(ec)

        def solo_red():
            dist.all_gather_into_tensor(qqg, qq)

        t_red = medir(solo_red)
        t_con = medir(concurrente, n=10)
        p(f"       [concurrencia pura] GEMM {t_g8:.0f} + red {t_red:.0f} = {t_g8+t_red:.0f} "
          f"-> juntos {t_con:.0f} us   (ideal {max(t_g8,t_red):.0f}, "
          f"solape real {100*(t_g8+t_red-t_con)/min(t_g8,t_red):.0f}%)")

        t_ser = medir(serie, n=10)
        t_p2 = medir(pipe(2), n=10)
        t_p4 = medir(pipe(4), n=10)
        t_p8 = medir(pipe(8), n=10)
        t_n2 = medir(pipe_nccl(2), n=10)
        t_n4 = medir(pipe_nccl(4), n=10)
        t_n8 = medir(pipe_nccl(8), n=10)
        mejor = min(t_p2, t_p4, t_p8, t_n2, t_n4, t_n8)
        techo = (t_g8 + t_ari) / max(t_g8, t_ari)
        p(f"{M:6d} {t_gemm:8.0f} {t_g8:8.0f} {t_arf:8.0f} {t_ari:8.0f} | "
          f"{t_ser:8.0f} {t_p4:8.0f} {t_n2:8.0f} {t_n4:8.0f} {t_n8:8.0f} "
          f"{t_ser/mejor:6.2f}x {techo:5.2f}x")

    # verificacion: el pipeline tiene que dar lo mismo que la serie
    M = 2048
    x = torch.randn(M, I_RANK, device=dev, dtype=torch.float16)
    xq = (x * 40).round().clamp(-127, 127).to(torch.int8)
    esc_x = torch.full((M, 1), 1 / 40.0, device=dev, dtype=torch.float32)
    pq = peso_q.t().contiguous().t()
    gq = lambda a, b: vops.cutlass_scaled_mm(a, pq, b, esc_w, torch.float16)
    ref = ar_int8(gq(xq, esc_x), w)
    paso = M // 4
    trozos = []
    for i in range(4):
        trozos.append(ar_int8(gq(xq[i * paso:(i + 1) * paso], esc_x[i * paso:(i + 1) * paso]), w))
    torch.cuda.synchronize()
    d = (torch.cat(trozos) - ref).abs().max().item()
    p(f"\nverificacion trozos vs entero: max dif = {d:.2e}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
