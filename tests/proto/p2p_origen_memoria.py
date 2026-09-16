"""¿De donde tiene que salir la memoria para que el P2P ande a pleno?

Tres formas de compartir el MISMO tensor entre procesos:
  A) torch reduce_tensor                         (lo que hacia p2p_buzon.py)
  B) cudaIpcGetMemHandle sobre el data_ptr de un tensor de torch
  C) cudaMalloc crudo + cudaIpcGetMemHandle

Interesa el ancho de banda del memcpyPeer y si un kernel propio puede desreferenciar el puntero.
"""
import ctypes, os, pickle, warnings, torch
warnings.filterwarnings("ignore")
import torch.distributed as dist, torch.multiprocessing.reductions as mpr
from vllm._genesis.kernels.ptx_lab import Kernel

class IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]

rank = int(os.environ["RANK"]); torch.cuda.set_device(rank)
dist.init_process_group("nccl"); gcpu = dist.new_group(backend="gloo"); w = dist.get_world_size()
dev = f"cuda:{rank}"; otro = 1 - rank
rt = ctypes.CDLL("libcudart.so")
rt.cudaIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), IpcHandle, ctypes.c_uint]
rt.cudaIpcGetMemHandle.argtypes = [ctypes.POINTER(IpcHandle), ctypes.c_void_p]
rt.cudaDeviceEnablePeerAccess(ctypes.c_int(otro), ctypes.c_uint(0))
def p(*a):
    if rank == 0: print(*a, flush=True)
N = 40 << 20

def cambiar_handles(ptr):
    h = IpcHandle(); assert rt.cudaIpcGetMemHandle(ctypes.byref(h), ctypes.c_void_p(ptr)) == 0
    mio = torch.frombuffer(bytearray(ctypes.string_at(ctypes.byref(h), 64)), dtype=torch.uint8).clone()
    tg = [torch.zeros(64, dtype=torch.uint8) for _ in range(w)]
    dist.all_gather(tg, mio, group=gcpu)
    ho = IpcHandle(); ctypes.memmove(ctypes.byref(ho), bytes(tg[otro].numpy()), 64)
    po = ctypes.c_void_p()
    rc = rt.cudaIpcOpenMemHandle(ctypes.byref(po), ho, ctypes.c_uint(1))
    return rc, po

def medir(f, n=15):
    for _ in range(4): f()
    torch.cuda.synchronize(); dist.barrier()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record()
    for _ in range(n): f()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b)/n*1000

# Un acceso ilegal ENVENENA el contexto CUDA: despues falla todo. Por eso cada variante se corre
# en su propio proceso, elegida con VARIANTE=A|B|C.
VAR = os.environ.get("VARIANTE", "C")
k = Kernel("sk21_p2p.cu","sk21_tirar",defs=["-DBLOQUES=8","-DHILOS=256"],warps=8); k.cargar()
local = torch.zeros(N, dtype=torch.uint8, device=dev)
p(f"{'forma de compartir':>42} {'memcpyPeer GB/s':>16} {'kernel puede leer':>18}")

if VAR == "A":
  # A) reduce_tensor
  t_a = torch.full((N,), rank+1, dtype=torch.uint8, device=dev)
  d = pickle.dumps(mpr.reduce_tensor(t_a)); buf = torch.frombuffer(bytearray(d), dtype=torch.uint8)
  n_ = torch.tensor([buf.numel()], dtype=torch.int64); ns=[torch.zeros(1,dtype=torch.int64) for _ in range(w)]
  dist.all_gather(ns, n_, group=gcpu); mx=int(max(int(x.item()) for x in ns))
  pad=torch.zeros(mx,dtype=torch.uint8); pad[:buf.numel()]=buf
  rec=[torch.zeros(mx,dtype=torch.uint8) for _ in range(w)]; dist.all_gather(rec,pad,group=gcpu)
  f_,a_ = pickle.loads(bytes(rec[otro][:int(ns[otro].item())].numpy())); t_otro = f_(*a_)
  t = medir(lambda: rt.cudaMemcpyPeerAsync(ctypes.c_void_p(t_otro.data_ptr()), ctypes.c_int(otro),
    ctypes.c_void_p(t_a.data_ptr()), ctypes.c_int(rank), ctypes.c_size_t(N),
    ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)))
  try:
    k.lanzar((8,1),[local, ctypes.c_uint64(t_otro.data_ptr()), N//16], 0); torch.cuda.synchronize(); ok="si"
  except Exception: ok="NO (ilegal)"
  p(f"{'A) torch reduce_tensor':>42} {N/t/1e3:16.2f} {ok:>18}")

if VAR == "B":
    # B) IPC a mano sobre el data_ptr de un tensor de torch
    t_b = torch.full((N,), rank+1, dtype=torch.uint8, device=dev)
    rc, po = cambiar_handles(t_b.data_ptr())
    if rc != 0:
       p(f"{'B) IPC a mano sobre tensor de torch':>42} {'(IpcGetMemHandle/Open falla: '+str(rc)+')':>16}")
    else:
       t = medir(lambda: rt.cudaMemcpyPeerAsync(ctypes.c_void_p(po.value), ctypes.c_int(otro),
           ctypes.c_void_p(t_b.data_ptr()), ctypes.c_int(rank), ctypes.c_size_t(N),
           ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)))
       try:
           k.lanzar((8,1),[local, ctypes.c_uint64(po.value), N//16], 0); torch.cuda.synchronize(); ok="si"
       except Exception: ok="NO (ilegal)"
       p(f"{'B) IPC a mano sobre tensor de torch':>42} {N/t/1e3:16.2f} {ok:>18}")

if VAR == "C":
    # C) cudaMalloc crudo
    ptr_c = ctypes.c_void_p()
    assert rt.cudaMalloc(ctypes.byref(ptr_c), ctypes.c_size_t(N)) == 0
    assert rt.cudaMemset(ptr_c, ctypes.c_int(rank+1), ctypes.c_size_t(N)) == 0
    rc, po_c = cambiar_handles(ptr_c.value)
    t = medir(lambda: rt.cudaMemcpyPeerAsync(ctypes.c_void_p(po_c.value), ctypes.c_int(otro),
       ctypes.c_void_p(ptr_c.value), ctypes.c_int(rank), ctypes.c_size_t(N),
       ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)))
    try:
       k.lanzar((8,1),[local, ctypes.c_uint64(po_c.value), N//16], 0); torch.cuda.synchronize(); ok="si"
    except Exception: ok="NO (ilegal)"
    p(f"{'C) cudaMalloc crudo':>42} {N/t/1e3:16.2f} {ok:>18}")
dist.destroy_process_group()
