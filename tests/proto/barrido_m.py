import sys, torch
sys.path.insert(0,"/usr/local/lib/python3.12/dist-packages")
sys.path.insert(0,"/w/tests/proto")
from banco_lmhead import medir
from vllm.scalar_type import scalar_types
# forma de un lineal GORDO del modelo: gate_up de una capa, por GPU
N, H, G = 17408, 5120, 128
dev="cuda"
b_q=torch.randint(-(2**31),2**31-1,(H//16,N*16//8),dtype=torch.int32,device=dev)
b_s=torch.ones((H//G,N),dtype=torch.float16,device=dev)*0.01
ws=torch.zeros(N//64*16,dtype=torch.int32,device=dev)
vacio=torch.empty(0,dtype=torch.int32,device=dev)
nb=N*H//2+(H//G)*N*2
print(f"gate_up por GPU: {N}x{H} int4 = {nb/1e6:.0f} MB")
print(f"{'M':>5}{'us':>9}{'GB/s':>9}{'us/token':>10}{'vs M=5':>9}")
base=None
for M in (1,5,10,20,40,60,80,120):
    x=torch.randn(M,H,dtype=torch.float16,device=dev)*0.1
    xq=(x*127).round().clamp(-127,127).to(torch.int8)
    a_s=torch.full((M,1),1/127,dtype=torch.float32,device=dev)
    gs=torch.ones(1,dtype=torch.float32,device=dev)
    try:
        t=medir(lambda: torch.ops._C.marlin_gemm(xq,None,b_q,None,b_s,a_s,None,None,vacio,vacio,ws,
                scalar_types.uint4b8.id,M,N,H,True,False,True,False), 30)
        if M==5: base=t
        r=f"{t/base:8.2f}x" if base else ""
        print(f"{M:5d}{t:9.1f}{nb/t/1e3:9.1f}{t/M:10.2f}{r:>9}")
    except Exception as e:
        print(f"{M:5d}  falla: {str(e)[:400]}")
