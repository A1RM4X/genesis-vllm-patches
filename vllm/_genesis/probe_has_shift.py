import torch
import vllm._genesis.kernels.sk01_gdn_qkvz as k1
import vllm._genesis.kernels.sk06_mlp_down as k6

sh = torch.rand((40, 64), dtype=torch.float32, device="cuda") * 0.01 + 1e-3
print("sk01._has_shift(no-cero) =", k1._has_shift(sh), " (deberia ser True)")
print("sk06._has_shift(no-cero) =", k6._has_shift(sh))

# La colision: un tensor de ceros, se libera, y otro no-cero cae en la MISMA
# direccion. El cache por data_ptr devuelve el False del muerto.
z = torch.zeros((40, 64), dtype=torch.float32, device="cuda")
p = z.data_ptr()
print("\nceros en", hex(p), "->", k1._has_shift(z))
del z
torch.cuda.empty_cache()
nz = torch.rand((40, 64), dtype=torch.float32, device="cuda") * 0.01 + 1e-3
print("no-cero en", hex(nz.data_ptr()), "misma direccion:", nz.data_ptr() == p)
print("sk01._has_shift(no-cero) =", k1._has_shift(nz), " <- deberia ser True")
print("sk06._has_shift(no-cero) =", k6._has_shift(nz))
