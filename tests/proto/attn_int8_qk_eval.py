"""Error de la atencion con q.k calculado en INT8 (estilo SageAttention) sobre q/k/v reales."""
import math, torch, sys
sys.path.insert(0, "/tmp")
import kv_dicc as E          # reusa carga y metrica de kv_diccionario_eval
dev = "cuda"; D, HKV, HQ = 256, 2, 12

def q8(x, dim=-1):
    s = x.abs().amax(dim, keepdim=True).clamp_min(1e-6) / 127
    return (x / s).round().clamp(-127, 127) * s

def q4(x, dim=-1):
    s = x.abs().amax(dim, keepdim=True).clamp_min(1e-6) / 7
    return (x / s).round().clamp(-7, 7) * s

def q4g(x, g=32):
    # int4 simetrico con escala por grupo de g elementos a lo largo de head_dim
    sh = x.shape; y = x.reshape(*sh[:-1], sh[-1] // g, g)
    s = y.abs().amax(-1, keepdim=True).clamp_min(1e-6) / 7
    return ((y / s).round().clamp(-7, 7) * s).reshape(sh)

def salida(q, k, v, pos, fq=None, fk=None):
    outs = []; g = HQ // HKV
    for lo in range(0, len(pos), 4):
        p = pos[lo:lo + 4]; T = int(p.max()) + 1
        qq = q[p].to(dev).float().view(len(p), HKV, g, D)
        kk = k[:T]
        if fq: qq = fq(qq)
        if fk: kk = fk(kk)
        sc = torch.einsum("bkgd,tkd->bkgt", qq, kk) / math.sqrt(D)
        mask = torch.arange(T, device=dev)[None, :] > p.to(dev)[:, None]
        w = sc.masked_fill(mask[:, None, None, :], float("-inf")).softmax(-1)
        outs.append(torch.einsum("bkgt,tkd->bkgd", w, v[:T]).reshape(len(p), HQ, D))
    return torch.cat(outs)

for capa in (3, 35):
    q, k, v = E.cargar(capa)
    N = k.shape[0]; pos = torch.linspace(16000, N - 1, 192).long()
    ref = salida(q, k, v, pos)
    media_k = k[:7488].mean(0, keepdim=True)            # suavizado: media por canal (calibrada en tramo 1)
    casos = {
        "q int8 + k int8 por token-cabeza": (lambda x: q8(x), lambda x: q8(x)),
        "q int8 + k int8 suavizado (-media)": (lambda x: q8(x), lambda x: q8(x - media_k) + media_k),
        "q int8 + k int8 con Hadamard": (lambda x: E.desrot(q8(E.rot(x))), lambda x: E.desrot(q8(E.rot(x)))),
        "q int4 + k int4 por token-cabeza": (lambda x: q4(x), lambda x: q4(x)),
        "q int4 + k int4 con Hadamard": (lambda x: E.desrot(q4(E.rot(x))), lambda x: E.desrot(q4(E.rot(x)))),
        "q int4 + k int4 Hadamard grupo 32": (lambda x: E.desrot(q4g(E.rot(x))), lambda x: E.desrot(q4g(E.rot(x)))),
        "q int8 + k int4 con Hadamard": (lambda x: E.desrot(q8(E.rot(x))), lambda x: E.desrot(q4(E.rot(x)))),
        "(ref) KV fp8 escala 1, q fp16": (None, lambda x: x.to(torch.float8_e4m3fn).float()),
    }
    print(f"\ncapa {capa}")
    for nom, (fq, fk) in casos.items():
        o = salida(q, k, v, pos, fq, fk)
        err = ((o - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-6)).mean().item()
        print(f"  {nom:38} error salida {100*err:.3f}%")
