"""Escalera de casos para SK-22: se agrega UNA cosa por vez hasta que rompe.

El caso minimo (1 tile de K, M=1, 1 grupo) ya daba exacto, asi que el problema esta en algo que
aparece al escalar: mas tiles de K (el pipeline), mas filas de M, o mas de un grupo de escalas.
"""
import sys
import torch
sys.path.insert(0, "/w/tests/proto")
from vllm._genesis.kernels.ptx_lab import Kernel
from sk22_banco import empaquetar

dev, TN, G = "cuda", 64, 128
k22 = Kernel("sk22_gemm_w4a8.cu", "sk22_gemm_w4a8", defs=[f"-DTN={TN}"], warps=4)


def caso(M, N, K, nombre, aleatorio=False, factor=1.0):
    torch.manual_seed(3)
    if aleatorio:
        a = (torch.randn(M, K, device=dev) * 20).round().clamp(-127, 127).to(torch.int8)
        q = torch.randint(0, 16, (K, N), dtype=torch.int32, device=dev)
        esc = torch.randint(-500, 500, (max(K // G, 1), N), dtype=torch.int16, device=dev)
    else:
        a = torch.ones(M, K, dtype=torch.int8, device=dev)
        q = torch.ones(K, N, dtype=torch.int32, device=dev)
        esc = torch.ones(max(K // G, 1), N, dtype=torch.int16, device=dev)
    ng = max(K // G, 1)
    sumas = a.reshape(M, ng, -1).sum(2, dtype=torch.int32).t().contiguous()
    a_esc = torch.ones(M, dtype=torch.float32, device=dev)
    b = empaquetar(q, TN)
    c = torch.zeros(M, N, dtype=torch.float16, device=dev)
    shmem = 3 * (16 * 128 + 4 * (TN // 8) * 32 * 4)
    k22.lanzar((N // TN, 1), [a, b, esc, sumas, a_esc, c, M, N, K, factor], shared=shmem)
    torch.cuda.synchronize()
    # repetir la MISMA llamada: si el resultado cambia hay registros sin inicializar o una
    # carrera. Es el chequeo que destapo el bug de `corr` en Marlin.
    primero = c.clone()
    inestable = 0.0
    for _ in range(3):
        c.zero_()
        k22.lanzar((N // TN, 1), [a, b, esc, sumas, a_esc, c, M, N, K, factor], shared=shmem)
        torch.cuda.synchronize()
        inestable = max(inestable, float((c.float() - primero.float()).abs().max()))
    c = primero
    # referencia exacta
    ref = torch.zeros(M, N, dtype=torch.float64, device=dev)
    for g in range(ng):
        aa = a[:, g * G:(g + 1) * G].double() if K >= G else a.double()
        qq = (q[g * G:(g + 1) * G].double() - 8.0) if K >= G else (q.double() - 8.0)
        ref += (aa @ qq) * esc[g].double().unsqueeze(0)
    ref *= factor
    d = float((c.double() - ref).abs().max())
    m = float(ref.abs().max())
    # fp16 tiene 11 bits de mantisa: el redondeo de la salida es el unico error admisible
    ok = d == 0 or d / max(m, 1e-9) < 2 ** -10
    print(f"  {nombre:<34} M={M:3d} N={N:5d} K={K:5d}  max|dif|={d:10.1f}  "
          f"rel={d/max(m,1):.2e}  {'OK' if ok else 'MAL'}"
          f"   {'estable' if inestable == 0 else f'CARRERA {inestable}'}")


caso(1, 64, 32, "1 tile K, M=1")
caso(1, 64, 64, "2 tiles K (pipeline)")
caso(1, 64, 128, "1 grupo completo")
caso(1, 64, 256, "2 grupos de escalas")
caso(4, 64, 128, "M=4")
caso(16, 64, 128, "M=16 (tile lleno)")
caso(4, 128, 128, "N=128 (2 bloques)")
caso(4, 128, 256, "todo junto")
caso(4, 128, 256, "todo junto, ALEATORIO", aleatorio=True, factor=1 / 4096)
caso(16, 256, 512, "grande, ALEATORIO", aleatorio=True, factor=1 / 4096)
caso(3, 192, 384, "M/N/K no potencias de 2", aleatorio=True, factor=1 / 4096)
