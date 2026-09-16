"""Cuanto cuesta cada pieza del intercambio P2P: la copia grande, la chica y la bandera.

El intercambio completo de PN136 daba 7.700 us para 42 MB (5,5 GB/s) cuando la copia sola va a
11,4 GB/s. Este banco parte el costo en pedazos para ver cual es el caro.
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
    rt, drv = ctypes.CDLL("libcudart.so"), ctypes.CDLL("libcuda.so")
    s = torch.cuda.current_stream()

    def peer(dst, src, n, stream=None):
        st = stream or s
        r = rt.cudaMemcpyPeerAsync(ctypes.c_void_p(dst), ctypes.c_int(otro),
                                   ctypes.c_void_p(src), ctypes.c_int(rank),
                                   ctypes.c_size_t(n), ctypes.c_void_p(st.cuda_stream))
        if r != 0:
            raise RuntimeError(f"memcpyPeer {r}")

    def wv(ptr, val, stream=None):
        st = stream or s
        return drv.cuStreamWriteValue32_v2(ctypes.c_void_p(st.cuda_stream),
                                           ctypes.c_ulonglong(ptr), ctypes.c_uint(val),
                                           ctypes.c_uint(0))

    def p(*a):
        if rank == 0:
            print(*a, flush=True)

    # cudaDeviceEnablePeerAccess: SIN esto cudaMemcpyPeerAsync igual funciona, pero por un camino
    # por etapas que va a la mitad de velocidad. El copy_ de torch lo habilita solo, por eso en
    # p2p_ancho.py la medicion salia rapida (venia despues de un copy_) y aca sale lenta.
    r_peer = rt.cudaDeviceEnablePeerAccess(ctypes.c_int(otro), ctypes.c_uint(0))
    p(f"cudaDeviceEnablePeerAccess({otro}) -> {r_peer}  (0 = ok, 704 = ya estaba)")

    def medir(f, n=30):
        for _ in range(8):
            f()
        torch.cuda.synchronize(); dist.barrier()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(n):
            f()
        b.record(); torch.cuda.synchronize()
        return a.elapsed_time(b) / n * 1000

    N = 40 * MB
    src = torch.zeros(N, dtype=torch.uint8, device=dev)
    dst = torch.zeros(N, dtype=torch.uint8, device=dev)
    dst_r = compartir(dst, gcpu, rank, w)
    band = torch.zeros(64, dtype=torch.int32, device=dev)
    band_r = compartir(band, gcpu, rank, w)

    p("costo de cada pieza (sin ninguna espera, solo el envio):")
    p(f"{'operacion':>34} {'us':>9} {'GB/s':>8}")
    for nombre, nb in (("copia de 40 MB", N), ("copia de 10 MB", 10 * MB),
                       ("copia de 320 KB (escalas)", 320 << 10), ("copia de 4 B (bandera)", 4)):
        t = medir(lambda nb=nb: peer(dst_r[otro].data_ptr(), src.data_ptr(), nb))
        p(f"{nombre:>34} {t:9.1f} {nb/t/1e3:8.2f}")

    # cuantas copias chicas entran en el costo de una grande
    t1 = medir(lambda: [peer(dst_r[otro].data_ptr() + i * 4, src.data_ptr(), 4) for i in range(8)])
    p(f"{'8 banderas seguidas':>34} {t1:9.1f}")

    # ¿se puede escribir un valor directo en la memoria del otro? (esperar NO se puede)
    r = wv(band_r[otro].data_ptr(), 7)
    torch.cuda.synchronize()
    p(f"\ncuStreamWriteValue32 sobre memoria del OTRO rank -> {'OK' if r == 0 else f'falla {r}'}")
    if r == 0:
        dist.barrier(); torch.cuda.synchronize()
        p(f"   lo escrito llego: {int(band[0].item()) == 7}")
        t = medir(lambda: wv(band_r[otro].data_ptr(), 9))
        p(f"   cuesta {t:.1f} us (contra {medir(lambda: peer(band_r[otro].data_ptr(), src.data_ptr(), 4)):.1f} us la copia de 4 B)")

    # el patron completo de PN136: datos + escalas + bandera, N veces
    p("\npatron de PN136 (datos + escalas + bandera por trozo, sin esperas):")
    for n in (2, 4, 8):
        paso = N // n
        def f(n=n, paso=paso):
            for i in range(n):
                peer(dst_r[otro].data_ptr() + i * paso, src.data_ptr(), paso)
                peer(dst_r[otro].data_ptr() + i * paso, src.data_ptr(), paso // 128)
                peer(band_r[otro].data_ptr() + 4 * i, src.data_ptr(), 4)
        t = medir(f, n=15)
        p(f"   {n} trozos: {t:8.1f} us  ({N/t/1e3:.2f} GB/s de datos)")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
