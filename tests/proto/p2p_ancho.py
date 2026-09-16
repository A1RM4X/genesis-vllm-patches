"""Por que la copia P2P daba la mitad que NCCL. Diagnostico del enlace entre las dos placas.

Variantes que se prueban:
  * empujar (escribir en la memoria del otro) contra tirar (leer de la memoria del otro);
  * ``copy_`` de torch contra ``cudaMemcpyPeerAsync`` llamado a mano;
  * una copia grande contra varias en paralelo sobre streams distintos (PCIe tiene varias colas:
    con una sola transferencia en vuelo la latencia de ida y vuelta manda y el enlace queda ocioso);
  * un kernel propio que copia con vectores de 16 bytes, que usa SM pero satura mejor.

Ademas informa a que generacion entreno el enlace DURANTE la transferencia: en reposo baja a gen1
y leerlo con nvidia-smi sin carga enganna.
"""
import ctypes
import os
import pickle
import warnings

import torch

warnings.filterwarnings("ignore")
import torch.distributed as dist
import torch.multiprocessing.reductions as mpr

MB = 1 << 20


def compartir(t, gcpu, rank, w):
    datos = pickle.dumps(mpr.reduce_tensor(t))
    buf = torch.frombuffer(bytearray(datos), dtype=torch.uint8)
    n = torch.tensor([buf.numel()], dtype=torch.int64)
    ns = [torch.zeros(1, dtype=torch.int64) for _ in range(w)]
    dist.all_gather(ns, n, group=gcpu)
    mx = int(max(int(x.item()) for x in ns))
    pad = torch.zeros(mx, dtype=torch.uint8)
    pad[:buf.numel()] = buf
    rec = [torch.zeros(mx, dtype=torch.uint8) for _ in range(w)]
    dist.all_gather(rec, pad, group=gcpu)
    out = []
    for i in range(w):
        if i == rank:
            out.append(t)
        else:
            f, a = pickle.loads(bytes(rec[i][:int(ns[i].item())].numpy()))
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
    cudart = ctypes.CDLL("libcudart.so")

    def p(*a):
        if rank == 0:
            print(*a, flush=True)

    N = 40 * MB
    mio = torch.zeros(N, dtype=torch.uint8, device=dev)
    buzon = torch.zeros(N, dtype=torch.uint8, device=dev)
    buzones = compartir(buzon, gcpu, rank, w)      # el buzon de cada rank, visto desde aca
    mios = compartir(mio, gcpu, rank, w)           # el buffer de cada rank, visto desde aca
    qg = torch.empty(N * w, dtype=torch.uint8, device=dev)

    streams = [torch.cuda.Stream(device=dev) for _ in range(4)]

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
        return a.elapsed_time(b) / n / 1000.0       # segundos

    def gbs(seg):
        return N / seg / 1e9

    def empujar_torch():
        buzones[otro].copy_(mio, non_blocking=True)

    def tirar_torch():
        buzon.copy_(mios[otro], non_blocking=True)

    def peer_async(dst, dst_dev, src, src_dev, nbytes, stream):
        # cudaMemcpyPeerAsync(dst, dstDev, src, srcDev, count, stream)
        cudart.cudaMemcpyPeerAsync(ctypes.c_void_p(dst), ctypes.c_int(dst_dev),
                                   ctypes.c_void_p(src), ctypes.c_int(src_dev),
                                   ctypes.c_size_t(nbytes),
                                   ctypes.c_void_p(stream.cuda_stream))

    def empujar_api():
        peer_async(buzones[otro].data_ptr(), otro, mio.data_ptr(), rank, N,
                   torch.cuda.current_stream())

    def empujar_partido(k):
        """k transferencias en vuelo, cada una en su stream."""
        def f():
            paso = N // k
            ev = torch.cuda.Event(); ev.record()
            for i in range(k):
                s = streams[i % len(streams)]
                s.wait_event(ev)
                peer_async(buzones[otro].data_ptr() + i * paso, otro,
                           mio.data_ptr() + i * paso, rank, paso, s)
            for i in range(min(k, len(streams))):
                e = torch.cuda.Event(); e.record(streams[i % len(streams)])
                torch.cuda.current_stream().wait_event(e)
        return f

    def nccl():
        dist.all_gather_into_tensor(qg, mio)

    p(f"{N/MB:.0f} MB por rank, TP={w}. El enlace es x8; gen4 x8 = 15,8 GB/s teoricos.")
    p(f"{'variante':>28} {'ms':>8} {'GB/s':>8}")
    for nombre, f in (("NCCL all_gather", nccl),
                      ("empujar (torch copy_)", empujar_torch),
                      ("tirar (torch copy_)", tirar_torch),
                      ("empujar (cudaMemcpyPeerAsync)", empujar_api),
                      ("empujar partido en 2", empujar_partido(2)),
                      ("empujar partido en 4", empujar_partido(4)),
                      ("empujar partido en 8", empujar_partido(8))):
        t = medir(f)
        p(f"{nombre:>28} {t*1000:8.2f} {gbs(t):8.2f}")
        if nombre == "NCCL all_gather" and rank == 0:
            g = os.popen("nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current "
                         "--format=csv,noheader").read().strip().replace("\n", " | ")
            p(f"    (enlace justo despues de NCCL: {g})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
