"""¿Un kernel propio puede leer la memoria de la otra placa?

Con un tensor de torch compartido por ``reduce_tensor`` el kernel da acceso ilegal, aunque el
``copy_`` de torch sobre el mismo puntero anda (ese usa el motor de copia, no un kernel). La
sospecha es el allocator de torch. Aca se prueba el camino crudo, que es el que usa el all-reduce
propio de vLLM: ``cudaMalloc`` + ``cudaIpcGetMemHandle`` + ``cudaIpcOpenMemHandle``.
"""
import ctypes
import os
import warnings

import torch

warnings.filterwarnings("ignore")
import torch.distributed as dist

from vllm._genesis.kernels.ptx_lab import Kernel

rank = int(os.environ["RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl")
gcpu = dist.new_group(backend="gloo")
w = dist.get_world_size()
otro = 1 - rank
dev = f"cuda:{rank}"
rt = ctypes.CDLL("libcudart.so")


def p(*a):
    if rank == 0:
        print(*a, flush=True)


r = rt.cudaDeviceEnablePeerAccess(ctypes.c_int(otro), ctypes.c_uint(0))
p(f"enablePeerAccess -> {r}  (0 ok, 704 ya estaba)")

N = 1 << 20          # 1 MB

class IpcHandle(ctypes.Structure):
    """cudaIpcMemHandle_t. TIENE que ser un struct: cudaIpcOpenMemHandle lo recibe POR VALOR, y
    ctypes pasa los arrays por referencia (de ahi el cudaErrorInvalidValue)."""
    _fields_ = [("reserved", ctypes.c_char * 64)]


rt.cudaIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), IpcHandle, ctypes.c_uint]
rt.cudaIpcGetMemHandle.argtypes = [ctypes.POINTER(IpcHandle), ctypes.c_void_p]

# 1) memoria cruda con cudaMalloc
ptr = ctypes.c_void_p()
assert rt.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(N)) == 0
assert rt.cudaMemset(ptr, ctypes.c_int(rank + 1), ctypes.c_size_t(N)) == 0

# 2) su handle de IPC (64 bytes)
handle = IpcHandle()
assert rt.cudaIpcGetMemHandle(ctypes.byref(handle), ptr) == 0

# 3) se intercambian los handles por el grupo de CPU
# OJO: leer el campo c_char*64 como atributo lo TRUNCA en el primer NUL. Hay que sacar los 64
# bytes crudos con string_at.
crudo = ctypes.string_at(ctypes.byref(handle), 64)
mio = torch.frombuffer(bytearray(crudo), dtype=torch.uint8).clone()
todos = [torch.zeros(64, dtype=torch.uint8) for _ in range(w)]
dist.all_gather(todos, mio, group=gcpu)

# 4) se abre el del otro
h_otro = IpcHandle()
ctypes.memmove(ctypes.byref(h_otro), bytes(todos[otro].numpy()), 64)
ptr_otro = ctypes.c_void_p()
CUDA_IPC_LAZY = 1
rc = rt.cudaIpcOpenMemHandle(ctypes.byref(ptr_otro), h_otro, ctypes.c_uint(CUDA_IPC_LAZY))
p(f"cudaIpcOpenMemHandle -> {rc}")
assert rc == 0

dst = torch.zeros(N, dtype=torch.uint8, device=dev)
rt.cudaDeviceSynchronize()
dist.barrier()
torch.cuda.synchronize()

# 5) correccion: el kernel lee de la memoria CRUDA del otro y escribe en un tensor local
k = Kernel("sk21_p2p.cu", "sk21_simple", defs=[], warps=4)
k.cargar()
k.lanzar(((N // 4 + 127) // 128, 1), [dst, ctypes.c_uint64(ptr_otro.value), N // 4], 0)
torch.cuda.synchronize()
p(f"5) kernel leyendo memoria CRUDA de la otra placa: {bool((dst == otro + 1).all())}")

# 6) ancho de banda del kernel propio segun cuantos SM use
GRANDE = 40 << 20
ptr_g = ctypes.c_void_p()
assert rt.cudaMalloc(ctypes.byref(ptr_g), ctypes.c_size_t(GRANDE)) == 0
assert rt.cudaMemset(ptr_g, ctypes.c_int(rank + 1), ctypes.c_size_t(GRANDE)) == 0
hg = IpcHandle()
assert rt.cudaIpcGetMemHandle(ctypes.byref(hg), ptr_g) == 0
mg = torch.frombuffer(bytearray(ctypes.string_at(ctypes.byref(hg), 64)), dtype=torch.uint8).clone()
tg = [torch.zeros(64, dtype=torch.uint8) for _ in range(w)]
dist.all_gather(tg, mg, group=gcpu)
hg_otro = IpcHandle()
ctypes.memmove(ctypes.byref(hg_otro), bytes(tg[otro].numpy()), 64)
pg_otro = ctypes.c_void_p()
assert rt.cudaIpcOpenMemHandle(ctypes.byref(pg_otro), hg_otro, ctypes.c_uint(1)) == 0
dg = torch.zeros(GRANDE, dtype=torch.uint8, device=dev)
n16 = GRANDE // 16
# el memset del buffer grande es asincrono: sin esto se puede leer al otro antes de que lo llene
rt.cudaDeviceSynchronize()
dist.barrier()
torch.cuda.synchronize()


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


qg = torch.empty(GRANDE * w, dtype=torch.uint8, device=dev)
t = medir(lambda: dist.all_gather_into_tensor(qg, dg))
p("")
p(f"6) referencia NCCL all_gather: {t:.0f} us = {GRANDE/t/1e3:.2f} GB/s")
t = medir(lambda: rt.cudaMemcpyPeerAsync(
    ctypes.c_void_p(pg_otro.value), ctypes.c_int(otro), ctypes.c_void_p(ptr_g.value),
    ctypes.c_int(rank), ctypes.c_size_t(GRANDE),
    ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)))
p(f"   referencia motor DMA:      {t:.0f} us = {GRANDE/t/1e3:.2f} GB/s")
# la referencia DMA EMPUJA, o sea que dejo el buffer del otro con nuestros datos. Hay que
# rellenarlo antes de comprobar exactitud, o el chequeo da falso negativo.
assert rt.cudaMemset(ptr_g, ctypes.c_int(rank + 1), ctypes.c_size_t(GRANDE)) == 0
rt.cudaDeviceSynchronize()
dist.barrier()
torch.cuda.synchronize()
p("")
p(f"{'bloques':>8} {'us':>9} {'GB/s':>8} {'correcto':>9} {'% de 82 SM':>12}")
kern = {}
for nb in (2, 4, 8, 16, 32, 64, 128):
    kk = Kernel("sk21_p2p.cu", "sk21_tirar", defs=[f"-DBLOQUES={nb}", "-DHILOS=256"], warps=8)
    kk.cargar()
    kern[nb] = kk
    dg.zero_()
    torch.cuda.synchronize()
    kk.lanzar((nb, 1), [dg, ctypes.c_uint64(pg_otro.value), n16], 0)
    torch.cuda.synchronize()
    ok = bool((dg == otro + 1).all())
    if not ok and nb == 8:
        vals = torch.unique(dg)[:6].tolist()
        malos = int((dg != otro + 1).sum())
        p(f"    [debug] esperaba {otro+1}; valores {vals}; mal {malos} de {dg.numel()} "
          f"({100*malos/dg.numel():.2f}%)")
    t = medir(lambda kk=kk, nb=nb: kk.lanzar((nb, 1), [dg, ctypes.c_uint64(pg_otro.value), n16], 0))
    p(f"{nb:8d} {t:9.0f} {GRANDE/t/1e3:8.2f} {str(ok):>9} {100*nb/82:11.0f}%")

# 7) ¿solapa con un GEMM?
p("")
p("7) solape con un GEMM (lo que NCCL no da):")
K = 8192
A = torch.randn(K, K, device=dev, dtype=torch.float16)
B = torch.randn(K, K, device=dev, dtype=torch.float16)
comm = torch.cuda.Stream(device=dev)
gemm = lambda: torch.mm(A, B)
t_g = medir(gemm, n=10)
p(f"   GEMM solo: {t_g:.0f} us")
p(f"   {'transporte':>26} {'solo':>8} {'con GEMM':>10} {'solape':>8}")


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


casos = [("NCCL", lambda st: dist.all_gather_into_tensor(qg, dg))]
for nb in (4, 8, 16, 32):
    casos.append((f"SK-21 con {nb} bloques",
                  lambda st, nb=nb: kern[nb].lanzar(
                      (nb, 1), [dg, ctypes.c_uint64(pg_otro.value), n16], 0)))
for nombre, envio in casos:
    ts = medir(lambda: envio(torch.cuda.current_stream()), n=12)
    tc = medir(juntos(envio), n=10)
    p(f"   {nombre:>26} {ts:7.0f}u {tc:9.0f}u {100*(t_g+ts-tc)/min(t_g,ts):7.0f}%")

dist.destroy_process_group()
