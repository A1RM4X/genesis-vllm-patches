"""Barrido de BLOCK_M / TILE / num_stages del kernel unified_attention con las formas
del Qwen3.8, sobre una COPIA del modulo con esos parametros como globales."""
import importlib.util, sys, time, torch
import vllm.v1.attention.ops.triton_unified_attention as orig
src = open(orig.__file__).read()
src = src.replace("current_platform.is_device_capability_family(100)", "_G_TUNED")
src = src.replace("        BLOCK_M = 32\n", "        BLOCK_M = _G_BM\n")
src = src.replace("        launch_num_warps = 8\n        launch_num_stages = 2\n",
                  "        launch_num_warps = _G_W\n        launch_num_stages = _G_ST\n")
src = src.replace("        TILE_SIZE_PREFILL = 128\n", "        TILE_SIZE_PREFILL = _G_TILE\n")
assert src.count("_G_") == 5, src.count("_G_")
src = "_G_TUNED, _G_BM, _G_W, _G_ST, _G_TILE = False, 16, 4, 3, 32\n" + src
open("/tmp/tua_var.py", "w").write(src)   # Triton exige que los @jit vivan en un archivo
sys.path.insert(0, "/tmp")
import tua_var as tua

dev = "cuda"; dt = torch.float16
HQ, HKV, D, BS = 12, 2, 256, 832
def armar(ctx, nq):
    nb = (ctx + BS - 1) // BS + 1
    k = torch.randn(nb, BS, HKV, D, device=dev, dtype=dt); v = torch.randn_like(k)
    q = torch.randn(nq, HQ, D, device=dev, dtype=dt)
    return (q, k, v, torch.empty_like(q), torch.arange(nb, device=dev, dtype=torch.int32).view(1, -1),
            torch.tensor([0, nq], device=dev, dtype=torch.int32), torch.tensor([ctx], device=dev, dtype=torch.int32))
def correr(ctx, nq, reps, ref=None):
    q, k, v, out, bt, cu, sl = armar(ctx, nq)
    f = lambda: tua.unified_attention(q=q, k=k, v=v, out=out, cu_seqlens_q=cu, max_seqlen_q=nq,
            seqused_k=sl, max_seqlen_k=ctx, softmax_scale=D ** -0.5, causal=True, alibi_slopes=None,
            window_size=(-1, -1), block_table=bt, softcap=0, sinks=None, q_descale=None, k_descale=None, v_descale=None)
    f(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(reps): f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3, out.float().norm().item()

ctx, nq, reps = 14976, 7488, 2
variantes = []
for bm in ():
    for tile in (32, 64, 128):
        for st in (1, 2):
            variantes.append((f"BLOCK_M={bm} TILE={tile} stages={st}", True, bm, 8, st, tile))
base = None
for nom, tuned, bm, w, st, tile in variantes:
    tua._G_TUNED, tua._G_BM, tua._G_W, tua._G_ST, tua._G_TILE = tuned, bm, w, st, tile
    try:
        ms, nrm = correr(ctx, nq, reps)
        base = base or ms
        print(f"{nom:34} {ms:8.1f} ms  {base/ms:5.2f}x  (norma {nrm:.1f})", flush=True)
    except Exception as e:
        print(f"{nom:34}  no entra: {str(e)[:70]}", flush=True)
    torch.cuda.empty_cache()

# ── referencia: FlashAttention paginada (misma familia que el camino FlashInfer) ──
print("\n--- referencia y decode ---", flush=True)
try:
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    def fa(ctx, nq, reps):
        q, k, v, out, bt, cu, sl = armar(ctx, nq)
        cuk = torch.tensor([0, ctx], device=dev, dtype=torch.int32)
        f = lambda: flash_attn_varlen_func(q, k, v, max_seqlen_q=nq, cu_seqlens_q=cu, max_seqlen_k=ctx,
                                            seqused_k=sl, softmax_scale=D ** -0.5, causal=True, block_table=bt)
        f(); torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(reps): f()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1e3
    print(f"FlashAttention paginada decode 4q@57k {fa(57000, 4, 20):8.2f} ms", flush=True)
except Exception as e:
    print("FA no disponible:", str(e)[:120])
# ── 3D (paralelo sobre la KV) contra 2D con 1 query, y 4 requests x 1 query ──
def correr3d(ctx, nseq, nq_por, reps, usar3d, segs=16):
    tua._G_TUNED = False
    nb = (ctx + BS - 1) // BS + 1
    k = torch.randn(nb * nseq, BS, HKV, D, device=dev, dtype=dt); v = torch.randn_like(k)
    nq = nseq * nq_por
    q = torch.randn(nq, HQ, D, device=dev, dtype=dt); out = torch.empty_like(q)
    bt = torch.arange(nb * nseq, device=dev, dtype=torch.int32).view(nseq, nb)
    cu = torch.arange(0, nq + 1, nq_por, device=dev, dtype=torch.int32)
    sl = torch.full((nseq,), ctx, device=dev, dtype=torch.int32)
    thr = 64
    kw = {}
    if usar3d:
        kw = dict(seq_threshold_3D=thr, num_par_softmax_segments=segs,
                  softmax_segm_output=torch.empty(thr, HQ, segs, D, device=dev),
                  softmax_segm_max=torch.empty(thr, HQ, segs, device=dev),
                  softmax_segm_expsum=torch.empty(thr, HQ, segs, device=dev))
    f = lambda: tua.unified_attention(q=q, k=k, v=v, out=out, cu_seqlens_q=cu, max_seqlen_q=nq_por,
            seqused_k=sl, max_seqlen_k=ctx, softmax_scale=D ** -0.5, causal=True, alibi_slopes=None,
            window_size=(-1, -1), block_table=bt, softcap=0, sinks=None, q_descale=None, k_descale=None,
            v_descale=None, **kw)
    f(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(reps): f()
    torch.cuda.synchronize()
    r = (time.perf_counter() - t0) / reps * 1e3
    o = out.float().clone()
    del k, v, q; torch.cuda.empty_cache()
    return r, o
t2, o2 = correr3d(57000, 1, 1, 20, False)
t3, o3 = correr3d(57000, 1, 1, 20, True)
print(f"1 query @57k:  2D serie {t2:6.2f} ms   3D paralelo {t3:6.2f} ms   {t2/t3:4.1f}x   (misma salida: {torch.allclose(o2, o3, rtol=1e-2, atol=1e-3)})", flush=True)
t2b, _ = correr3d(57000, 1, 4, 20, False)
print(f"4 queries (MTP) @57k: 2D serie {t2b:6.2f} ms  (el 3D no acepta max_seqlen_q>1)", flush=True)
import sys; sys.exit(0)
dec = [("generico decode 4q@57k", False, 16, 4, 3, 32)]
for bm in (24, 32, 48):
    for tile in (16, 32, 64):
        for st in (1, 2):
            for w in (4, 8):
                dec.append((f"dec BM={bm} TILE={tile} st={st} w={w}", True, bm, w, st, tile))
for nom, tuned, bm, w, st, tile in dec:
    tua._G_TUNED, tua._G_BM, tua._G_W, tua._G_ST, tua._G_TILE = tuned, bm, w, st, tile
    try:
        ms, _ = correr(57000, 4, 20)
        print(f"{nom:38} {ms:8.2f} ms", flush=True)
    except Exception as e:
        print(f"{nom:38}  no entra: {str(e)[:50]}", flush=True)
    torch.cuda.empty_cache()
