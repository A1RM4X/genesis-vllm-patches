"""SK-20: tiempo acumulado por etapa (A, maximos, B, cola de la escala)."""
import torch
from vllm._genesis.kernels.ptx_lab import Kernel
dev = "cuda"; torch.manual_seed(0); torch.zeros(1, device=dev)
REP = 50
def medir(f, n=50, rep=REP):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): f()
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(rep): f()
    for _ in range(3): gr.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(n): gr.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b)/(n*rep)*1e3
M, K = 5, 5120
x = torch.randn(M, K, device=dev, dtype=torch.float16)
res = torch.randn(M, K, device=dev, dtype=torch.float16)
w = (1 + 0.1*torch.randn(K, device=dev, dtype=torch.float16))
g = torch.tensor([0.000173], device=dev, dtype=torch.float32)
r_out = torch.empty(M, K, dtype=torch.float16, device=dev)
q_out = torch.empty(M, K, dtype=torch.int8, device=dev)
esc = torch.empty(M, dtype=torch.float32, device=dev)
nom = {1: "A (suma+claves)", 2: "+ maximos", 3: "+ pasada B", 4: "+ cola escala"}
for hilos in (256, 512):
    for et in (1, 2, 3, 4):
        for hay in (1, 0):
            k = Kernel("sk20_norm_quant.cu", "sk20_norm_quant",
                       defs=[f"-DHILOS={hilos}", f"-DETAPA={et}"], warps=hilos//32)
            k.cargar()
            args = [x.view(torch.int16), res.view(torch.int16), w.view(torch.int16),
                    g.view(torch.int32), r_out.view(torch.int16), q_out, esc.view(torch.int32),
                    K, x.stride(0), hay]
            print(f"HILOS={hilos} res={hay} etapa {et} {nom[et]:>16}: "
                  f"{medir(lambda: k.lanzar((M,1), args, 4*K)):6.2f} us")
