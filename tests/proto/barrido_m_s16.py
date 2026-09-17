"""Mismo barrido pero sobre el Marlin propio (PN130), para ver si max_thread_m_blocks=8 borra
el acantilado de M=64."""
import sys, os, torch
sys.path.insert(0,"/usr/local/lib/python3.12/dist-packages")
sys.path.insert(0,"/w/tests/proto")
from banco_lmhead import medir
from vllm.scalar_type import scalar_types
torch.ops.load_library(os.environ["S16_SO"])
N,H,G=17408,5120,128; dev="cuda"
b_q=torch.randint(-(2**31),2**31-1,(H//16,N*16//8),dtype=torch.int32,device=dev)
b_s=(torch.ones((H//G,N),dtype=torch.float16,device=dev)*0.01)
m=b_s.abs().max(); b_s16=(b_s/m*4096).round().to(torch.int16).view(b_s.dtype)
ws=torch.zeros(N//64*16,dtype=torch.int32,device=dev)
vacio=torch.empty(0,dtype=torch.int32,device=dev)
gs=(m.float()/4096).reshape(1)
nb=N*H//2+(H//G)*N*2
print(f"gate_up {N}x{H} int4 = {nb/1e6:.0f} MB   ({os.path.basename(os.path.dirname(os.environ['S16_SO']))})")
print(f"{'M':>5}{'us':>9}{'GB/s':>9}{'vs M=5':>9}")
base=None
for M in (5,64,70,80,88,96,104,112,128,144,160,256):
    x=torch.randn(M,H,dtype=torch.float16,device=dev)*0.1
    xq=(x*127).round().clamp(-127,127).to(torch.int8)
    a_s=torch.full((M,1),1/127,dtype=torch.float32,device=dev)
    # PN140: las sumas van precalculadas — en produccion las emite el kernel que cuantiza la
    # activacion, que ya recorre A entera, asi que no entran en el tiempo del GEMM.
    sumas = (xq.to(torch.int32).reshape(M, H // G, G).sum(2).t().contiguous()
             if os.environ.get("PN140") == "1" else None)
    try:
        t=medir(lambda: torch.ops.genesis_marlin.marlin_gemm_s16(
            xq,None,b_q,None,b_s16,a_s,None,None,vacio,vacio,sumas,ws,
            scalar_types.uint4b8.id,M,N,H,True,False,True,False), 30)
        if M==5: base=t
        print(f"{M:5d}{t:9.1f}{nb/t/1e3:9.1f}{t/base:8.2f}x")
    except Exception as e:
        print(f"{M:5d}  falla: {str(e)[:80]}")
