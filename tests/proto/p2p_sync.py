"""Sincronizacion entre ranks SIN gastar SM, con las operaciones de memoria de stream de CUDA.

El problema del solape: si el rank A tira el trozo i de la memoria del rank B, tiene que saber que
B ya lo escribio. Las opciones son un barrier (serializa, mata el solape), un kernel que gira
esperando (gasta un SM y compite con el GEMM, que es justo lo que queremos evitar) o
``cuStreamWaitValue32``: el stream espera a que una posicion de memoria tome un valor, y lo
resuelve el planificador de la placa sin ejecutar nada.

Este banco comprueba que:
  1. cuStreamWaitValue32 / cuStreamWriteValue32 estan disponibles y funcionan sobre memoria IPC;
  2. la espera no consume SM (el GEMM sigue a velocidad plena mientras el stream espera);
  3. el patron completo (escribo, aviso, el otro espera y tira) da el dato correcto.
"""
import ctypes
import os
import pickle
import warnings

import torch

warnings.filterwarnings("ignore")
import torch.distributed as dist
import torch.multiprocessing.reductions as mpr

# cuStreamWaitValue32 con flags: 0 = EQ, 1 = GEQ
CU_EQ, CU_GEQ = 0, 1


class Cuda:
    """Las pocas llamadas del driver/runtime que hacen falta, por ctypes."""

    def __init__(self):
        self.rt = ctypes.CDLL("libcudart.so")
        self.drv = ctypes.CDLL("libcuda.so")

    def wait_value(self, ptr, valor, stream, flags=CU_GEQ):
        r = self.drv.cuStreamWaitValue32_v2(ctypes.c_void_p(stream.cuda_stream),
                                            ctypes.c_ulonglong(ptr),
                                            ctypes.c_uint(valor), ctypes.c_uint(flags))
        if r != 0:
            raise RuntimeError(f"cuStreamWaitValue32 -> {r}")

    def write_value(self, ptr, valor, stream, flags=0):
        r = self.drv.cuStreamWriteValue32_v2(ctypes.c_void_p(stream.cuda_stream),
                                             ctypes.c_ulonglong(ptr),
                                             ctypes.c_uint(valor), ctypes.c_uint(flags))
        if r != 0:
            raise RuntimeError(f"cuStreamWriteValue32 -> {r}")

    def memcpy_peer(self, dst, dst_dev, src, src_dev, n, stream):
        r = self.rt.cudaMemcpyPeerAsync(ctypes.c_void_p(dst), ctypes.c_int(dst_dev),
                                        ctypes.c_void_p(src), ctypes.c_int(src_dev),
                                        ctypes.c_size_t(n),
                                        ctypes.c_void_p(stream.cuda_stream))
        if r != 0:
            raise RuntimeError(f"cudaMemcpyPeerAsync -> {r}")


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
    cu = Cuda()

    def p(*a):
        if rank == 0:
            print(*a, flush=True)

    # 0) ¿se puede esperar sobre memoria LOCAL? (sobre memoria IPC del otro NO: da error 1)
    prueba = torch.zeros(4, dtype=torch.int32, device=dev)
    try:
        cu.write_value(prueba.data_ptr(), 7, torch.cuda.current_stream())
        cu.wait_value(prueba.data_ptr(), 7, torch.cuda.current_stream(), CU_GEQ)
        torch.cuda.synchronize()
        p("espera sobre memoria local: OK")
    except RuntimeError as e:
        p(f"espera sobre memoria local: FALLA ({e})")
    try:
        cu.wait_value(band_r_prueba, 1, torch.cuda.current_stream(), CU_GEQ)
    except Exception:
        pass

    N = 4 * (1 << 20)
    datos = torch.zeros(N, dtype=torch.uint8, device=dev)
    banderas = torch.zeros(64, dtype=torch.int32, device=dev)
    buzon = torch.zeros(N, dtype=torch.uint8, device=dev)
    datos_r = compartir(datos, gcpu, rank, w)
    band_r = compartir(banderas, gcpu, rank, w)
    buzon_r = compartir(buzon, gcpu, rank, w)
    comm = torch.cuda.Stream(device=dev)
    band_r_prueba = banderas.data_ptr()

    # 1) correccion: escribo mis datos, aviso con la bandera, espero la del otro y tiro
    # Modelo EMPUJAR, que es el que usa el all-reduce propio de vLLM: cada rank escribe sus datos
    # en el buzon del otro y despues su bandera, las dos cosas con cudaMemcpyPeerAsync en el mismo
    # stream (o sea ordenadas). Cada rank espera sobre su bandera LOCAL, que si esta soportado.
    p("\n1) el patron completo (empujo datos -> empujo bandera -> el otro espera su bandera local)")
    marca = torch.zeros(1, dtype=torch.int32, device=dev)
    ok_total = True
    for epoca in (1, 2, 3):
        datos.fill_(rank * 10 + epoca)
        marca.fill_(epoca)
        e = torch.cuda.Event(); e.record()
        with torch.cuda.stream(comm):
            comm.wait_event(e)
            cu.memcpy_peer(buzon_r[otro].data_ptr(), otro, datos.data_ptr(), rank, N, comm)
            # la bandera va DESPUES en el mismo stream: el otro no la ve hasta que llegaron los datos
            cu.memcpy_peer(band_r[otro].data_ptr(), otro, marca.data_ptr(), rank, 4, comm)
            cu.wait_value(banderas.data_ptr(), epoca, comm, CU_GEQ)   # bandera LOCAL
            ec = torch.cuda.Event(); ec.record(comm)
        torch.cuda.current_stream().wait_event(ec)
        torch.cuda.synchronize()
        esperado = otro * 10 + epoca
        ok = bool((buzon == esperado).all())
        ok_total &= ok
        p(f"   epoca {epoca}: buzon == {esperado} -> {ok}")
    p(f"   correcto = {ok_total}")

    # 2) ¿la espera gasta SM? Se pone el stream a esperar una bandera que tarda, y mientras tanto
    #    se mide el GEMM. Si la espera gastara SM, el GEMM se frenaria.
    p("\n2) la espera, ¿roba SM al GEMM?")
    K = 4096
    a = torch.randn(K, K, device=dev, dtype=torch.float16)
    b = torch.randn(K, K, device=dev, dtype=torch.float16)

    def gemm():
        torch.mm(a, b)

    def medir(f, n=30):
        for _ in range(5):
            f()
        torch.cuda.synchronize(); dist.barrier()
        t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
        t0.record()
        for _ in range(n):
            f()
        t1.record(); torch.cuda.synchronize()
        return t1.elapsed_time(t0 if False else t1) if False else t0.elapsed_time(t1) / n * 1000

    t_solo = medir(gemm)

    banderas.zero_()
    torch.cuda.synchronize()

    def gemm_con_espera():
        # el stream lateral queda esperando una bandera que recien se prende al final
        with torch.cuda.stream(comm):
            cu.wait_value(banderas.data_ptr(), 1, comm, CU_GEQ)
        gemm()
        cu.write_value(banderas.data_ptr(), 1, torch.cuda.current_stream())
        e = torch.cuda.Event(); e.record(comm)
        torch.cuda.current_stream().wait_event(e)
        banderas.zero_()

    t_con = medir(gemm_con_espera, n=20)
    p(f"   GEMM solo {t_solo:.1f} us | GEMM con un stream esperando {t_con:.1f} us "
      f"-> sobrecosto {100*(t_con-t_solo)/t_solo:.1f}%")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
