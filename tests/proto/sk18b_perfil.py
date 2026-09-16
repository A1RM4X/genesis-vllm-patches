import time, torch
from vllm._genesis.kernels.ptx_lab import Kernel
dev="cuda"; N=256; SH=2*(2*64*128+64*2*128)
LUT=torch.round(32639*torch.exp(-16.0*torch.arange(1024).float()/1023)).to(torch.int32).to(dev); LUT[-1]=0
M,Kr,CH=24,57000,1024; K=((Kr+CH-1)//CH)*CH
Sz=(torch.randn(M,K,device=dev)*2.5*65536).clamp(-(1<<26),1<<26).to(torch.int32); Sz[:,Kr:]=-(1<<28)
zmax=Sz.amax(1).to(torch.int32).contiguous(); mq=torch.full((M,),1023*65536//(16*65536),dtype=torch.int32,device=dev)
svf=torch.randint(16384,32767,(K,),dtype=torch.int16,device=dev); Vt=torch.randint(-127,128,(N,K),dtype=torch.int8,device=dev)
inv=torch.zeros(M,dtype=torch.int32,device=dev)
for d in (0,1,2):
    k=Kernel("sk18b_wv.cu","sk18b_wv",defs=[f"-DDIAG={d}"],warps=4)
    out=torch.zeros(K//CH,M,N,dtype=torch.int32,device=dev); lo=torch.zeros_like(out)
    g=((N+63)//64,((M+63)//64)*(K//CH))
    f=lambda: k.lanzar(g,[Sz,zmax,mq,inv,LUT,svf,Vt,out,lo,M,N,K,CH,K//CH],shared=SH)
    for _ in range(3): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(20): f()
    torch.cuda.synchronize(); print(f"DIAG={d}: {(time.perf_counter()-t)/20*1e3:.3f} ms",flush=True)
