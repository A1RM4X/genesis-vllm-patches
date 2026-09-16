"""Por que pierde TRITON_ATTN: kernel unified_attention con las formas del Qwen3.8
(12 q heads, 2 kv heads por rank, head 256, bloque 832) en prefill por chunks y en
decode spec (4 queries) sobre contexto largo. Compara el camino generico de Ampere
contra el afinado para head 256 que upstream solo habilita en Blackwell."""
import time, torch, sys
import vllm.v1.attention.ops.triton_unified_attention as tua
from vllm.platforms import current_platform

dev = "cuda"; dt = torch.float16
HQ, HKV, D, BS = 12, 2, 256, 832

def armar(ctx, nq):
    nb = (ctx + BS - 1) // BS + 1
    k = torch.randn(nb, BS, HKV, D, device=dev, dtype=dt)
    v = torch.randn(nb, BS, HKV, D, device=dev, dtype=dt)
    q = torch.randn(nq, HQ, D, device=dev, dtype=dt)
    out = torch.empty_like(q)
    bt = torch.arange(nb, device=dev, dtype=torch.int32).view(1, -1)
    cu = torch.tensor([0, nq], device=dev, dtype=torch.int32)
    sl = torch.tensor([ctx], device=dev, dtype=torch.int32)
    return q, k, v, out, bt, cu, sl

def correr(ctx, nq, reps):
    q, k, v, out, bt, cu, sl = armar(ctx, nq)
    f = lambda: tua.unified_attention(q=q, k=k, v=v, out=out, cu_seqlens_q=cu, max_seqlen_q=nq,
            seqused_k=sl, max_seqlen_k=ctx, softmax_scale=D ** -0.5, causal=True, alibi_slopes=None,
            window_size=(-1, -1), block_table=bt, softcap=0, sinks=None,
            q_descale=None, k_descale=None, v_descale=None)
    for _ in range(2): f()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(reps): f()
    torch.cuda.synchronize()
    r = (time.perf_counter() - t0) / reps * 1e3
    del q, k, v, out; torch.cuda.empty_cache()
    return r

casos = [("prefill chunk 7488 sobre ctx 14976", 14976, 7488, 2),
         ("prefill chunk 2048 sobre ctx 8192", 8192, 2048, 4),
         ("decode spec 4 q sobre ctx 57k", 57000, 4, 20)]
orig = current_platform.is_device_capability_family
for nombre, ctx, nq, reps in casos:
    current_platform.is_device_capability_family = orig
    a = correr(ctx, nq, reps)
    current_platform.is_device_capability_family = lambda fam, _o=orig: True if fam == 100 else _o(fam)
    b = correr(ctx, nq, reps)
    print(f"{nombre:38}: generico {a:8.1f} ms   afinado-head256 {b:8.1f} ms   {a/b:4.2f}x", flush=True)
current_platform.is_device_capability_family = orig
