import torch
from vllm._genesis.kernels.ptx_lab import Kernel
k = Kernel("sk22_sondac.cu", "sk22_sondac", warps=1)
out = torch.zeros(128, dtype=torch.int32, device="cuda")
k.lanzar((1, 1), [out]); torch.cuda.synchronize()
o = out.cpu().tolist()
print("C: (lane, i) -> fila   [valor/32 - 1]")
print(f"  {'lane':>5} {'c[0]':>6} {'c[1]':>6} {'c[2]':>6} {'c[3]':>6}   filas deducidas")
malos = 0
for L in range(0, 32, 4):
    fs = [o[L*4+i] // 32 - 1 for i in range(4)]
    esperado = [L // 4, L // 4, L // 4 + 8, L // 4 + 8]
    if fs != esperado: malos += 1
    print(f"  {L:5d} {o[L*4]:6d} {o[L*4+1]:6d} {o[L*4+2]:6d} {o[L*4+3]:6d}   {fs}  "
          f"{'ok' if fs == esperado else 'ESPERABA ' + str(esperado)}")
print(f"\n  la regla 'c[0],c[1] -> lane/4 ; c[2],c[3] -> lane/4+8' "
      f"{'SE CONFIRMA' if malos == 0 else 'ES FALSA'}")
