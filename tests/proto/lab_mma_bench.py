"""mma s8 m16n8k32 contra s4 m16n8k64: correccion del layout y TOPS de emision pura."""
import sys, time, torch
sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
from vllm._genesis.kernels.ptx_lab import Kernel, contar_instrucciones

dev = "cuda"
torch.manual_seed(0)


def empacar_s8(A, B):
    """A [G,16,32] row-major, B [G,32,8] col-major -> registros [G,32,4] y [G,32,2]."""
    G = A.shape[0]
    ra = A.reshape(G, 32, 16).contiguous().view(torch.int32).view(G, 32, 4).contiguous()
    rb = B.permute(0, 2, 1).reshape(G, 32, 8).contiguous().view(torch.int32).view(G, 32, 2).contiguous()
    return ra, rb


def nibbles_a_bytes(n, alto_primero=False):
    u = (n & 0xF).to(torch.uint8)
    lo, hi = (u[..., 1::2], u[..., 0::2]) if alto_primero else (u[..., 0::2], u[..., 1::2])
    return (lo | (hi << 4)).to(torch.uint8)


def empacar_s4(A, B, alto_primero=False):
    """A [G,16,64] row-major, B [G,64,8] col-major, valores -8..7."""
    G = A.shape[0]
    ba = nibbles_a_bytes(A.reshape(G, 1024), alto_primero)            # [G,512]
    bb = nibbles_a_bytes(B.permute(0, 2, 1).reshape(G, 512), alto_primero)  # [G,256]
    ra = ba.view(torch.int32).view(G, 32, 4).contiguous()
    rb = bb.view(torch.int32).view(G, 32, 2).contiguous()
    return ra, rb


def referencia(A, B):
    C = torch.bmm(A.to(torch.float64), B.to(torch.float64)).round()     # [G,16,8] exacto en f64
    return C.reshape(A.shape[0], 32, 4).to(torch.int32)


def probar(nombre, entrada, A, B, ra, rb):
    k = Kernel("lab_mma.cu", entrada, warps=1)
    G = A.shape[0]
    out = torch.zeros(G, 32, 4, dtype=torch.int32, device=dev)
    k.lanzar((G, 1), [ra, rb, out, 1], sync=True)
    ref = referencia(A, B)
    ok = torch.equal(out, ref)
    return k, ok, (out - ref).abs().max().item()


def tops(k, ra, rb, G, reps, ops_por_inst, it=5):
    out = torch.zeros(G, 32, 4, dtype=torch.int32, device=dev)
    k.lanzar((G, 1), [ra, rb, out, reps], sync=True)
    t0 = time.perf_counter()
    for _ in range(it):
        k.lanzar((G, 1), [ra, rb, out, reps])
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / it
    ops = G * reps * ops_por_inst
    return ops / dt / 1e12, dt


G = 4096
A8 = torch.randint(-127, 128, (G, 16, 32), dtype=torch.int8, device=dev)
B8 = torch.randint(-127, 128, (G, 32, 8), dtype=torch.int8, device=dev)
ra8, rb8 = empacar_s8(A8, B8)
k8, ok8, d8 = probar("s8", "lab_mma_s8", A8, B8, ra8, rb8)
print(f"s8 m16n8k32: correcto={ok8} (max dif {d8})", flush=True)

A4 = torch.randint(-8, 8, (G, 16, 64), dtype=torch.int8, device=dev)
B4 = torch.randint(-8, 8, (G, 64, 8), dtype=torch.int8, device=dev)
k4 = None
for alto in (False, True):
    ra4, rb4 = empacar_s4(A4, B4, alto)
    try:
        k4, ok4, d4 = probar("s4", "lab_mma_s4", A4, B4, ra4, rb4)
        print(f"s4 m16n8k64 nibble {'alto' if alto else 'bajo'} primero: correcto={ok4} (max dif {d4})", flush=True)
        if ok4:
            break
    except Exception as e:
        print(f"s4 fallo: {str(e)[:400]}", flush=True)
        break

print("instrucciones s8:", contar_instrucciones(k8.ptx()), flush=True)
if k4 is not None:
    print("instrucciones s4:", contar_instrucciones(k4.ptx()), flush=True)
    for reps in (64, 256):
        t8, dt8 = tops(k8, ra8, rb8, G, reps, 16 * 8 * 32)
        t4, dt4 = tops(k4, ra4, rb4, G, reps, 16 * 8 * 64)
        print(f"REPS={reps}: s8 {t8:7.1f} TOPS ({dt8*1e3:.1f} ms)   s4 {t4:7.1f} TOPS ({dt4*1e3:.1f} ms)   s4/s8 {t4/t8:.2f}x", flush=True)
