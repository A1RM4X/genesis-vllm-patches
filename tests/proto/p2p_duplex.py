"""¿La copia P2P baja a la mitad cuando las DOS placas mandan a la vez?"""
import ctypes, os, pickle, warnings, torch
warnings.filterwarnings("ignore")
import torch.distributed as dist, torch.multiprocessing.reductions as mpr
MB = 1 << 20
def compartir(t, g, rank, w):
    d = pickle.dumps(mpr.reduce_tensor(t)); buf = torch.frombuffer(bytearray(d), dtype=torch.uint8)
    n = torch.tensor([buf.numel()], dtype=torch.int64); ns = [torch.zeros(1, dtype=torch.int64) for _ in range(w)]
    dist.all_gather(ns, n, group=g); mx = int(max(int(x.item()) for x in ns))
    pad = torch.zeros(mx, dtype=torch.uint8); pad[:buf.numel()] = buf
    rec = [torch.zeros(mx, dtype=torch.uint8) for _ in range(w)]; dist.all_gather(rec, pad, group=g)
    return [t if i == rank else pickle.loads(bytes(rec[i][:int(ns[i].item())].numpy()))[0](
        *pickle.loads(bytes(rec[i][:int(ns[i].item())].numpy()))[1]) for i in range(w)]
rank = int(os.environ["RANK"]); torch.cuda.set_device(rank)
dist.init_process_group("nccl"); gcpu = dist.new_group(backend="gloo"); w = dist.get_world_size()
dev = f"cuda:{rank}"; otro = 1 - rank
rt = ctypes.CDLL("libcudart.so"); s = torch.cuda.current_stream()
rt.cudaDeviceEnablePeerAccess(ctypes.c_int(otro), ctypes.c_uint(0))
N = 40 * MB
src = torch.zeros(N, dtype=torch.uint8, device=dev); dst = torch.zeros(N, dtype=torch.uint8, device=dev)
dst_r = compartir(dst, gcpu, rank, w); src_r = compartir(src, gcpu, rank, w)
def push(): rt.cudaMemcpyPeerAsync(ctypes.c_void_p(dst_r[otro].data_ptr()), ctypes.c_int(otro),
    ctypes.c_void_p(src.data_ptr()), ctypes.c_int(rank), ctypes.c_size_t(N), ctypes.c_void_p(s.cuda_stream))
def pull(): rt.cudaMemcpyPeerAsync(ctypes.c_void_p(dst.data_ptr()), ctypes.c_int(rank),
    ctypes.c_void_p(src_r[otro].data_ptr()), ctypes.c_int(otro), ctypes.c_size_t(N), ctypes.c_void_p(s.cuda_stream))
def medir(f, n=20):
    for _ in range(5): f()
    torch.cuda.synchronize(); dist.barrier()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record()
    for _ in range(n): f()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b)/n*1000
def p(*a):
    if rank == 0: print(*a, flush=True)
nada = lambda: None
p(f"{'caso':>40} {'us':>9} {'GB/s':>8}")
for nom, f0, f1 in (("solo rank0 empuja (unidireccional)", push, nada),
                    ("solo rank0 tira  (unidireccional)", pull, nada),
                    ("los dos empujan a la vez", push, push),
                    ("los dos tiran a la vez", pull, pull),
                    ("rank0 empuja / rank1 tira", push, pull)):
    f = f0 if rank == 0 else f1
    t = medir(f)
    if rank == 1 and f is nada: t = float("nan")
    dist.barrier()
    p(f"{nom:>40} {t:9.1f} {N/t/1e3:8.2f}")
# NCCL de referencia
qg = torch.empty(N*w, dtype=torch.uint8, device=dev)
t = medir(lambda: dist.all_gather_into_tensor(qg, src))
p(f"{'NCCL all_gather (los dos, duplex)':>40} {t:9.1f} {N/t/1e3:8.2f}")
dist.destroy_process_group()
