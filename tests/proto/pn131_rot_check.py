import sys, types, os, torch
exec(open("/p/pn131_offline.py").read().split("q, k, v = A.cargar(35)")[0])
q, k, v = A.cargar(35)
N = 2000
nblocks = 6
kv = kv_nuevo(nblocks)
slots = torch.arange(N, device=dev)
P.escribir(impl, layer, k[:N].half(), v[:N].half(), kv, slots)
kd, vd = P.decuantizar_bloques(impl, kv, torch.tensor([0, 1, 2], device=dev))
Kd = kd.reshape(-1, 2, D)[:N].float()
s = P._signos_dev(torch.device(dev)).float(); H = P._hadamard(torch.device(dev), torch.float32)
Kr = torch.einsum("de,the->thd", H, k[:N].float() * s) / 16.0
err_rot = ((Kd - Kr).norm(dim=-1) / Kr.norm(dim=-1)).mean().item()
# que tan ortogonal es la rotacion torch
ang = ((Kr.norm(dim=-1) - k[:N].float().norm(dim=-1)).abs() / k[:N].float().norm(dim=-1)).mean().item()
# error de cuantizar int8 por token-cabeza K rotada vs sin rotar (en float, referencia)
def q8(x):
    sc = x.abs().amax(-1, keepdim=True) / 127
    return torch.round(x / sc) * sc
e_r = ((q8(Kr) - Kr).norm(dim=-1) / Kr.norm(dim=-1)).mean().item()
e_n = ((q8(k[:N].float()) - k[:N].float()).norm(dim=-1) / k[:N].float().norm(dim=-1)).mean().item()
print(f"K decuant(kernel rotado) vs H(sk)/16 torch: {100*err_rot:.3f}%  | norma preservada dif {100*ang:.4f}%")
print(f"error int8 ideal: K rotada {100*e_r:.3f}%  K sin rotar {100*e_n:.3f}%")
print("max|k|", k[:N].abs().max().item(), "max|Kr|", Kr.abs().max().item())
