"""Cuanto cuesta HOY la suma del residuo dentro de SK-20, y cuanto trafico mueve.

La propuesta era pasar el residual a int32 con exponente compartido para que la suma sea un add
entero. Esto mide la premisa: la diferencia entre HAY_RES=1 y HAY_RES=0 es exactamente el costo
de sumar el residuo (mas el de leerlo y escribirlo).
"""
import torch
from vllm._genesis.kernels.ptx_lab import Kernel
dev = "cuda"; torch.manual_seed(0); torch.zeros(1, device=dev)


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
    return a.elapsed_time(b) / (n * rep) * 1e3


K = 5120
print(f"{'M':>6} {'con res':>9} {'sin res':>9} {'costo':>8} {'% del kernel':>13} "
      f"{'GB/s con':>9} {'GB/s sin':>9}")
for M in (5, 40, 512, 8192):
    x = torch.randn(M, K, device=dev, dtype=torch.float16)
    res = torch.randn(M, K, device=dev, dtype=torch.float16)
    w = (1 + 0.1 * torch.randn(K, device=dev, dtype=torch.float16))
    g_ = torch.tensor([0.000173], device=dev, dtype=torch.float32)
    ro = torch.empty(M, K, dtype=torch.float16, device=dev)
    qo = torch.empty(M, K, dtype=torch.int8, device=dev)
    es = torch.empty(M, dtype=torch.float32, device=dev)
    t = {}
    for hay in (1, 0):
        mejor = None
        for h in (128, 256, 512, 1024):
            k = Kernel("sk20_norm_quant.cu", "sk20_norm_quant", defs=[f"-DHILOS={h}"], warps=h // 32)
            k.cargar()
            args = [x.view(torch.int16), res.view(torch.int16), w.view(torch.int16),
                    g_.view(torch.int32), ro.view(torch.int16), qo, es.view(torch.int32),
                    K, x.stride(0), hay]
            v = medir(lambda: k.lanzar((M, 1), args, 4 * K))
            mejor = v if mejor is None or v < mejor else mejor
        t[hay] = mejor
    # bytes: con residuo lee x+res+w y escribe res_out+q; sin residuo lee x+w y escribe q
    b_con = M * K * (2 + 2 + 2 + 1) + K * 2
    b_sin = M * K * (2 + 1) + K * 2
    print(f"{M:6d} {t[1]:8.2f}u {t[0]:8.2f}u {t[1]-t[0]:7.2f}u "
          f"{100*(t[1]-t[0])/t[1]:12.0f}% {b_con/t[1]/1e3:9.0f} {b_sin/t[0]/1e3:9.0f}")
