"""Valida vllm/_genesis/rot_qk.py con los q/k/v volcados (PN123): exactitud sin cuantizar
y error de salida con KV int8/int4 por token-cabeza, sin rotar / Hadamard / WUSH."""
import math, os, sys, torch
os.environ["GENESIS_PN126_ROT"] = "hadamard"
sys.path.insert(0, "/p")
import wush_qk_eval as E                      # cargar(), qn(), error()
from vllm._genesis import rot_qk as R
D, HKV, HQ, G = 256, 2, 12, 6

def aplicar(M, x, hkv):                        # x [T, h, D] -> por cabeza KV
    T = x.shape[0]
    xh = x.reshape(T, hkv, -1, D)
    return torch.einsum("gde,tgie->tgid", M.float(), xh).reshape(x.shape)

for capa in (3, 35):
    q, k, v, n1 = E.cargar(capa)
    pos = torch.linspace(n1 + 512, q.shape[0] - 1, 48).long()
    Mq = torch.einsum("tgid,tgie->gde", q[:n1].reshape(n1, HKV, G, D).double(), q[:n1].reshape(n1, HKV, G, D).double()) / (n1 * G)
    Mk = torch.einsum("tgd,tge->gde", k[:n1].double(), k[:n1].double()) / n1
    Tq, Tk = R.construir_wush(Mq, Mk)
    H = R._hadamard_aleatoria(D, "cpu").expand(HKV, D, D)
    casos = {"sin rotar": (torch.eye(D).expand(HKV, D, D), torch.eye(D).expand(HKV, D, D)),
             "hadamard": (H, H), "wush": (Tq, Tk)}
    print(f"\ncapa {capa}")
    for nom, (A, B) in casos.items():
        qr, kr = aplicar(A, q, HKV), aplicar(B, k, 1 * HKV)
        exacto = E.error(q, k, v, pos, lambda x: aplicar(A, x, HKV), lambda x: aplicar(B, x, HKV))
        e8 = E.error(q, k, v, pos, lambda x: aplicar(A, x, HKV), lambda x: E.qn(aplicar(B, x, HKV), 8, 256))
        e4 = E.error(q, k, v, pos, lambda x: aplicar(A, x, HKV), lambda x: E.qn(aplicar(B, x, HKV), 4, 64))
        f8 = E.error(q, k, v, pos, lambda x: aplicar(A, x, HKV), lambda x: aplicar(B, x, HKV).to(torch.float8_e4m3fn).float())
        print(f"  {nom:10s} sin cuantizar {exacto:.4f}% | KV fp8 {f8:.3f}% | KV int8 {e8:.3f}% | KV int4 g64 {e4:.3f}%", flush=True)
