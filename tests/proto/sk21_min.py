"""Prueba minima: ¿un kernel puede leer la memoria de la otra placa mapeada por IPC?"""
import ctypes, os, pickle, warnings, torch
warnings.filterwarnings("ignore")
import torch.distributed as dist, torch.multiprocessing.reductions as mpr
from vllm._genesis.kernels.ptx_lab import Kernel
def compartir(t, g, rank, w):
    d = pickle.dumps(mpr.reduce_tensor(t)); buf = torch.frombuffer(bytearray(d), dtype=torch.uint8)
    n = torch.tensor([buf.numel()], dtype=torch.int64); ns=[torch.zeros(1,dtype=torch.int64) for _ in range(w)]
    dist.all_gather(ns, n, group=g); mx=int(max(int(x.item()) for x in ns))
    pad=torch.zeros(mx,dtype=torch.uint8); pad[:buf.numel()]=buf
    rec=[torch.zeros(mx,dtype=torch.uint8) for _ in range(w)]; dist.all_gather(rec,pad,group=g)
    out=[]
    for i in range(w):
        if i==rank: out.append(t)
        else:
            f,a = pickle.loads(bytes(rec[i][:int(ns[i].item())].numpy())); out.append(f(*a))
    return out
rank=int(os.environ["RANK"]); torch.cuda.set_device(rank)
dist.init_process_group("nccl"); gcpu=dist.new_group(backend="gloo"); w=dist.get_world_size()
dev=f"cuda:{rank}"; otro=1-rank
rt=ctypes.CDLL("libcudart.so")
r=rt.cudaDeviceEnablePeerAccess(ctypes.c_int(otro), ctypes.c_uint(0))
def p(*a):
    if rank==0: print(*a, flush=True)
p(f"enablePeerAccess -> {r}")
N=1024
src=torch.full((N,), rank+1, dtype=torch.uint8, device=dev)
dst=torch.zeros(N, dtype=torch.uint8, device=dev)
src_r=compartir(src, gcpu, rank, w); dst_r=compartir(dst, gcpu, rank, w)
# 1) ¿torch puede leer la memoria del otro? (esto ya sabemos que anda)
dst.copy_(src_r[otro]); torch.cuda.synchronize()
p(f"1) copy_ tirando: {bool((dst==otro+1).all())}")
# 1b) alineacion de los punteros
p(f"1b) alineacion: local {src.data_ptr()%16}, remoto {src_r[otro].data_ptr()%16}")
# 1c) ¿un kernel tonto de 4 bytes puede leer de la otra placa?
ks=Kernel("sk21_p2p.cu","sk21_simple",defs=[],warps=4); ks.cargar()
dst.zero_(); torch.cuda.synchronize()
try:
    ks.lanzar((4,1),[dst, src_r[otro], N//4], 0)
    torch.cuda.synchronize()
    p(f"1c) kernel simple de 4 B leyendo remoto: {bool((dst==otro+1).all())}")
except Exception as e:
    p(f"1c) kernel simple de 4 B leyendo remoto: FALLA {type(e).__name__}")
    os._exit(1)
# 2) ¿un kernel propio puede LEER de la otra placa?
k=Kernel("sk21_p2p.cu","sk21_tirar",defs=["-DBLOQUES=2","-DHILOS=256"],warps=8); k.cargar()
dst.zero_(); torch.cuda.synchronize()
try:
    k.lanzar((2,1),[dst, src_r[otro], N//16]); torch.cuda.synchronize()
    p(f"2) kernel tirando (lee remoto): {bool((dst==otro+1).all())}")
except Exception as e:
    p(f"2) kernel tirando: FALLA {type(e).__name__}")
    os._exit(1)
# 3) ¿un kernel propio puede ESCRIBIR en la otra placa?
dist.barrier(); torch.cuda.synchronize()
try:
    k2=Kernel("sk21_p2p.cu","sk21_empujar",defs=["-DBLOQUES=2","-DHILOS=256"],warps=8); k2.cargar()
    k2.lanzar((2,1),[dst_r[otro], src, N//16]); torch.cuda.synchronize(); dist.barrier()
    p(f"3) kernel empujando (escribe remoto): {bool((dst==otro+1).all())}")
except Exception as e:
    p(f"3) kernel empujando: FALLA {type(e).__name__}")
dist.destroy_process_group()
