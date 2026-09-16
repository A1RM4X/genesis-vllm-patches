"""PN120 en int4: cuanto error de mas cuesta bajar el all-reduce de int8 a int4.

En prefill el all-reduce es el 41,6% del tiempo de GPU y va a 10,8 GB/s, o sea al techo del PCIe:
lo unico que queda es mandar menos bytes. Hoy son int8 con escala por grupo de 64 (1,03 bytes por
valor contando las escalas). En int4 con grupo 16 serian 0,625 bytes: 0,61x, ~16% del prefill.

Este script repite la tabla del docstring de ``ar_int8`` con las mismas condiciones (outliers
dispersos, que es el regimen realista) y agrega:
  * int4 simetrico y ASIMETRICO (con cero, que es lo que usamos en la KV int4),
  * la rotacion de Hadamard, que conmuta con la suma y se des-rota una sola vez,
  * la cadena de 64 capas con residual y RMSNorm, para ver si el error se acumula.

El error que se reporta es el mismo del modulo: ||suma_cuantizada - suma_exacta|| / ||suma_exacta||
"""
import math
import torch

dev = "cuda"
torch.manual_seed(0)
M, H, W = 4096, 5120, 2          # tokens, ancho, ranks


def parciales(m=M, h=H, w=W):
    """Parciales de un all-reduce con outliers dispersos, como las activaciones reales."""
    xs = []
    for _ in range(w):
        x = torch.randn(m, h, device=dev, dtype=torch.float16)
        idx = torch.randint(0, h, (m, max(1, h // 1000)), device=dev)
        x.scatter_(1, idx, x.gather(1, idx) * 12)
        xs.append(x)
    return xs


def cuant_sim(x, g, bits):
    """Simetrica por grupo: escala = amax/qmax."""
    m, h = x.shape
    q = (1 << (bits - 1)) - 1
    v = x.view(m, h // g, g).float()
    s = v.abs().amax(-1, keepdim=True).clamp_min(1e-6) / q
    return (v / s).round().clamp(-q, q) * s


def cuant_asim(x, g, bits):
    """Asimetrica por grupo: escala = (max-min)/(2^bits - 1) y cero entero."""
    m, h = x.shape
    n = (1 << bits) - 1
    v = x.view(m, h // g, g).float()
    mx = v.amax(-1, keepdim=True)
    mn = v.amin(-1, keepdim=True)
    s = ((mx - mn) / n).clamp_min(1e-6)
    z = (-mn / s).round()
    return ((v / s + z).round().clamp(0, n) - z) * s


def hadamard(n):
    """Hadamard de tamano potencia de 2. H=5120 no lo es (1024*5), asi que se aplica por bloques."""
    h = torch.ones(1, 1, device=dev)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(h.shape[0])).half()


def rotar(x, h):
    """Hadamard por bloques de h.shape[0] a lo ancho de la fila."""
    b = h.shape[0]
    return (x.view(x.shape[0], -1, b) @ h).view(x.shape)


def error(xs, f, rot=None):
    """f cuantiza un parcial; la suma se hace despues, como en el all-reduce real."""
    exacta = sum(x.float() for x in xs)
    if rot is not None:
        acc = sum(f(rotar(x, rot)).view(x.shape) for x in xs)
        aprox = rotar(acc.half(), rot.t()).float()
    else:
        acc = sum(f(x).view(x.shape) for x in xs)
        aprox = acc
    return ((aprox - exacta).norm() / exacta.norm()).item()


def bytes_por_valor(g, bits):
    return bits / 8 + 2 / g                    # escalas fp16, una por grupo


xs = parciales()
print(f"parciales: {W} x [{M}, {H}] con outliers dispersos")
print(f"{'esquema':>22} {'grupo':>6} {'error':>9} {'bytes/valor':>12} {'vs int8 g64':>11}")
base = bytes_por_valor(64, 8)
filas = []
for bits in (8, 4):
    for g in (128, 64, 32, 16, 8):
        for nombre, f in (("simetrica", cuant_sim), ("asimetrica", cuant_asim)):
            if bits == 8 and nombre == "asimetrica":
                continue
            e = error(xs, lambda x, g=g, b=bits, f=f: f(x, g, b))
            b = bytes_por_valor(g, bits) + (2 / g if nombre == "asimetrica" else 0)
            filas.append((f"int{bits} {nombre}", g, e, b))
            print(f"int{bits} {nombre:>12} {g:6d} {e:9.5f} {b:12.3f} {b/base:10.2f}x")

rot = hadamard(1024)
for bits, g in ((4, 32), (4, 16)):
    e = error(xs, lambda x, g=g, b=bits: cuant_sim(x, g, b), rot=rot)
    b = bytes_por_valor(g, bits)
    print(f"int{bits} simetrica+Hadamard {g:6d} {e:9.5f} {b:12.3f} {b/base:10.2f}x")

# ¿se acumula a traves de las capas? Cadena con residual + RMSNorm, 64 capas.
print("\ncadena de 16 capas (residual + RMSNorm, 2 all-reduce por capa):")
for etiqueta, g, bits, f in (("int8 g64", 64, 8, cuant_sim),
                             ("int4 g32", 32, 4, cuant_sim),
                             ("int4 g16", 16, 4, cuant_sim),
                             ("int4 g16 asim", 16, 4, cuant_asim)):
    torch.manual_seed(1)
    h_ex = torch.randn(M, H, device=dev, dtype=torch.float16).float()
    h_ap = h_ex.clone()
    for capa in range(16):
        for _ in range(2):
            ps = parciales(M, H, W)
            ex = sum(p.float() for p in ps)
            ap = sum(f(p, g, bits).view(p.shape) for p in ps)
            h_ex = h_ex + ex
            h_ap = h_ap + ap
            h_ex = h_ex / h_ex.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
            h_ap = h_ap / h_ap.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
    print(f"  {etiqueta:>14}: {((h_ap - h_ex).norm() / h_ex.norm()).item():9.5f}")
