"""SK-15b (volcado desfasado): exactitud contra la referencia de SK-15 y tiempo."""
import time, torch
from vllm._genesis.kernels.ptx_lab import Kernel
from vllm._genesis.kernels import sk15_w4a4g as S
SH = 2 * (256 * 128 + 64 * 2 * 128) + 2 * 4 * (256 + 2 * 64)
def tras4(t):
    """[R, NG] -> [NG, R rellenado a multiplo de 4] contiguo."""
    R = t.shape[0]; Rp = (R + 3) & ~3
    return torch.nn.functional.pad(t, (0, 0, 0, Rp - R)).t().contiguous()

def lanzar(k, an, sa, gn, sg, un, su, M, N, Kb, tras=False):
    out = torch.empty(M, N, dtype=torch.float16, device="cuda")
    if tras:
        sa, sg, su = tras4(sa), tras4(sg), tras4(su)
    k.lanzar(((N + 63) // 64, (M + 255) // 256), [an, gn, un, sa, sg, su, out, M, N, Kb], shared=SH)
    return out
kb = {e: Kernel("sk15b_gateup_w4a4g.cu", "sk15b_gateup_w4a4g", defs=[f"-DESC_SHARED={e}"], warps=32) for e in (0, 1)}
# exactitud
for (M, K, N) in [(300, 1024, 130), (2048, 5120, 1024)]:
    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda") * 3; Wg = torch.randn(N, K, device="cuda") * 0.02; Wu = torch.randn(N, K, device="cuda") * 0.02
    qa, sa = S.cuantizar(S.rotar(x)); qg, sg = S.cuantizar(S.rotar(Wg)); qu, su = S.cuantizar(S.rotar(Wu))
    ref = S.referencia(qa, sa, qg, sg, qu, su)
    for d, k in kb.items():
        o = lanzar(k, S.nibbles(qa), sa, S.nibbles(qg), sg, S.nibbles(qu), su, M, N, K // 2, tras=bool(d)).float()
        print(f"  M={M} K={K} N={N} ESC_SHARED={d}: err {((o-ref).norm()/ref.norm()).item():.2e}", flush=True)
# tiempo
M, K, N = 7488, 5120, 8704
torch.manual_seed(0)
x = torch.randn(M, K, device="cuda") * 3; an, sa = S.preparar_act(x); del x
W = torch.randn(N, K, device="cuda") * 0.02; gn, sg = S.preparar_pesos(W); un, su = S.preparar_pesos(W); del W
saT, sgT, suT = tras4(sa), tras4(sg), tras4(su)
casos = {"SK-15b escalas global": lambda: lanzar(kb[0], an, sa, gn, sg, un, su, M, N, K // 2),
         "SK-15b escalas en shared": lambda: lanzar(kb[1], an, saT, gn, sgT, un, suT, M, N, K // 2)}
k14 = Kernel("sk14_gateup_w4a4.cu", "sk14_gateup_w4a4", defs=["-DS4=1"], warps=8)
sa1, sg1, su1 = sa.mean(1).contiguous(), sg.mean(1).contiguous(), su.mean(1).contiguous()
casos["SK-14 (sin grupos)"] = lambda: lanzar(k14, an, sa1, gn, sg1, un, su1, M, N, K // 2)
for f in casos.values():
    for _ in range(2): f()
torch.cuda.synchronize(); res = {k: [] for k in casos}
for _ in range(5):
    for k, f in casos.items():
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(3): f()
        torch.cuda.synchronize(); res[k].append((time.perf_counter() - t0) / 3)
for k, v in res.items():
    print(f"  {k:24s} {sorted(v)[2]*1e3:8.2f} ms", flush=True)
