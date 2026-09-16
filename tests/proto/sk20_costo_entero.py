"""Cuanto cuesta hacer la suma del residuo en enteros en vez de con el sumador fp16 del hardware."""
import torch
from vllm._genesis.kernels.ptx_lab import Kernel
dev="cuda"; torch.manual_seed(0); torch.zeros(1, device=dev)
def medir(f, n=50, rep=50):
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3): f()
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(rep): f()
    for _ in range(3): g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(n): g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b)/(n*rep)*1e3
for M, K in ((5,5120), (40,5120), (8192,5120)):
    x = torch.randn(M,K,device=dev,dtype=torch.float16)
    res = torch.randn(M,K,device=dev,dtype=torch.float16)
    w = (1+0.1*torch.randn(K,device=dev,dtype=torch.float16))
    g_ = torch.tensor([0.000173],device=dev,dtype=torch.float32)
    ro = torch.empty(M,K,dtype=torch.float16,device=dev)
    qo = torch.empty(M,K,dtype=torch.int8,device=dev)
    es = torch.empty(M,dtype=torch.float32,device=dev)
    ref = None
    out = []
    for etiqueta, defs in (("entero", []), ("add.f16", ["-DFPADD=1"])):
        mejor=None
        for h in (256, 512, 1024):
            k = Kernel("sk20_norm_quant.cu","sk20_norm_quant",defs=[f"-DHILOS={h}"]+defs,warps=h//32)
            k.cargar()
            args=[x.view(torch.int16),res.view(torch.int16),w.view(torch.int16),g_.view(torch.int32),
                  ro.view(torch.int16),qo,es.view(torch.int32),K,x.stride(0),1]
            t=medir(lambda: k.lanzar((M,1),args,4*K))
            if mejor is None or t<mejor[0]: mejor=(t,h)
            if etiqueta=="add.f16" and h==256:
                k.lanzar((M,1),args,4*K); torch.cuda.synchronize()
                ref=(ro.clone(),qo.clone())
        out.append(mejor)
    # comprobar que dan lo mismo
    k = Kernel("sk20_norm_quant.cu","sk20_norm_quant",defs=["-DHILOS=256"],warps=8); k.cargar()
    args=[x.view(torch.int16),res.view(torch.int16),w.view(torch.int16),g_.view(torch.int32),
          ro.view(torch.int16),qo,es.view(torch.int32),K,x.stride(0),1]
    k.lanzar((M,1),args,4*K); torch.cuda.synchronize()
    igual = bool((ro.view(torch.int16)==ref[0].view(torch.int16)).all()) and bool((qo==ref[1]).all())
    print(f"M={M:5d}: entero {out[0][0]:7.2f}us (h={out[0][1]})  add.f16 {out[1][0]:7.2f}us "
          f"(h={out[1][1]})  sobrecosto {out[0][0]-out[1][0]:5.2f}us  bits iguales={igual}")
