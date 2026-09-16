"""SK-21: la copia entre placas escrita a mano, contra el motor DMA y contra NCCL.

Las dos preguntas que decide este banco:
  1. ¿cuanto ancho de banda saca un kernel propio con POCOS bloques?
  2. ¿cuanto solapa con un GEMM? (que es lo que NCCL no da: se toma todos los SM que quiere)

Uso: torchrun --nproc_per_node=2 sk21_p2p_test.py
"""
import ctypes
import os
import pickle
import warnings

import torch

warnings.filterwarnings("ignore")
import torch.distributed as dist
import torch.multiprocessing.reductions as mpr

from vllm._genesis.kernels.ptx_lab import Kernel

MB = 1 << 20


def compartir(t, gcpu, rank, w):
    d = pickle.dumps(mpr.reduce_tensor(t))
    buf = torch.frombuffer(bytearray(d), dtype=torch.uint8)
    n = torch.tensor([buf.numel()], dtype=torch.int64)
    ns = [torch.zeros(1, dtype=torch.int64) for _ in range(w)]
    dist.all_gather(ns, n, group=gcpu)
    mx = int(max(int(x.item()) for x in ns))
    pad = torch.zeros(mx, dtype=torch.uint8)
    pad[: buf.numel()] = buf
    rec = [torch.zeros(mx, dtype=torch.uint8) for _ in range(w)]
    dist.all_gather(rec, pad, group=gcpu)
    out = []
    for i in range(w):
        if i == rank:
            out.append(t)
        else:
            f, a = pickle.loads(bytes(rec[i][: int(ns[i].item())].numpy()))
            out.append(f(*a))
    return out


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    gcpu = dist.new_group(backend="gloo")
    w = dist.get_world_size()
    dev = f"cuda:{rank}"
    otro = 1 - rank
    rt = ctypes.CDLL("libcudart.so")
    rt.cudaDeviceEnablePeerAccess(ctypes.c_int(otro), ctypes.c_uint(0))
    s = torch.cuda.current_stream()
    comm = torch.cuda.Stream(device=dev)

    def p(*a):
        if rank == 0:
            print(*a, flush=True)

    N = 40 * MB
    n16 = N // 16
    src = torch.arange(N, dtype=torch.uint8, device=dev) if rank == 0 else \
        torch.full((N,), 7, dtype=torch.uint8, device=dev)
    src.fill_(rank + 1)
    dst = torch.zeros(N, dtype=torch.uint8, device=dev)
    src_r = compartir(src, gcpu, rank, w)
    dst_r = compartir(dst, gcpu, rank, w)

    def medir(f, n=20):
        for _ in range(5):
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

    def dma_push():
        rt.cudaMemcpyPeerAsync(ctypes.c_void_p(dst_r[otro].data_ptr()), ctypes.c_int(otro),
                               ctypes.c_void_p(src.data_ptr()), ctypes.c_int(rank),
                               ctypes.c_size_t(N), ctypes.c_void_p(s.cuda_stream))

    qg = torch.empty(N * w, dtype=torch.uint8, device=dev)

    p(f"{N/MB:.0f} MB por rank. Referencias: NCCL y el motor DMA.")
    p(f"{'variante':>38} {'us':>9} {'GB/s':>8} {'correcto':>9}")
    t = medir(lambda: dist.all_gather_into_tensor(qg, src))
    p(f"{'NCCL all_gather':>38} {t:9.1f} {N/t/1e3:8.2f}")
    t = medir(dma_push)
    p(f"{'cudaMemcpyPeerAsync (motor DMA)':>38} {t:9.1f} {N/t/1e3:8.2f}")
    t = medir(lambda: dst.copy_(src_r[otro], non_blocking=True))
    torch.cuda.synchronize()
    ok = bool((dst == otro + 1).all())
    p(f"{'copy_ de torch, tirando':>38} {t:9.1f} {N/t/1e3:8.2f} {str(ok):>9}")

    p("\nkernel propio (SK-21), tirando, segun cuantos bloques use:")
    p(f"{'bloques':>8} {'us':>9} {'GB/s':>8} {'correcto':>9} {'% de los 82 SM':>15}")
    kern = {}
    for nb in (2, 4, 8, 16, 32, 64):
        k = Kernel("sk21_p2p.cu", "sk21_tirar", defs=[f"-DBLOQUES={nb}", "-DHILOS=256"], warps=8)
        k.cargar()
        kern[nb] = k
        dst.zero_()
        f = lambda k=k, nb=nb: k.lanzar((nb, 1), [dst, src_r[otro], n16])
        f()
        torch.cuda.synchronize()
        ok = bool((dst == otro + 1).all())
        t = medir(f)
        p(f"{nb:8d} {t:9.1f} {N/t/1e3:8.2f} {str(ok):>9} {100*nb/82:14.0f}%")

    # ¿solapa con un GEMM?
    p("\nsolape con un GEMM (lo que NCCL no da):")
    K = 8192
    a = torch.randn(K, K, device=dev, dtype=torch.float16)
    b = torch.randn(K, K, device=dev, dtype=torch.float16)

    def gemm():
        torch.mm(a, b)

    t_g = medir(gemm, n=10)
    p(f"   GEMM solo: {t_g:.0f} us")
    p(f"{'transporte':>30} {'solo':>9} {'con GEMM':>10} {'solape':>8}")

    def juntos(envio):
        def f():
            e = torch.cuda.Event(); e.record()
            with torch.cuda.stream(comm):
                comm.wait_event(e)
                envio(comm)
                ec = torch.cuda.Event(); ec.record(comm)
            gemm()
            torch.cuda.current_stream().wait_event(ec)
        return f

    casos = [("NCCL", lambda st: dist.all_gather_into_tensor(qg, src)),
             ("motor DMA", lambda st: rt.cudaMemcpyPeerAsync(
                 ctypes.c_void_p(dst_r[otro].data_ptr()), ctypes.c_int(otro),
                 ctypes.c_void_p(src.data_ptr()), ctypes.c_int(rank),
                 ctypes.c_size_t(N), ctypes.c_void_p(st.cuda_stream)))]
    for nb in (4, 8, 16):
        casos.append((f"SK-21 con {nb} bloques",
                      lambda st, nb=nb: kern[nb].lanzar((nb, 1), [dst, src_r[otro], n16],
                                                        sync=False)))
    for nombre, envio in casos:
        t_solo = medir(lambda: envio(s), n=15)
        t_con = medir(juntos(envio), n=10)
        sol = 100 * (t_g + t_solo - t_con) / min(t_g, t_solo)
        p(f"{nombre:>30} {t_solo:8.0f}u {t_con:9.0f}u {sol:7.0f}%")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
