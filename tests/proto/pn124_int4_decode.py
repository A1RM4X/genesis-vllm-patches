"""Tile de decode del kernel INT4 en el camino 3D (decode spec aplanado) @57k."""
import os, time, torch
os.environ["GENESIS_ENABLE_PN124_TRITON_AMPERE"] = "1"
import vllm._genesis.triton_attn_ampere as g
import vllm.v1.attention.ops.int4_per_token_head as i4
dev = "cuda"; dt = torch.float16
HQ, HKV, D, BS = 12, 2, 256, 1616
torch.manual_seed(0)
ctx, N, Q = 57000, 1, 4
nb = ctx // BS + 2
kc = torch.zeros(nb, BS, HKV, D // 2, device=dev, dtype=torch.uint8); vc = torch.zeros_like(kc)
ksc = torch.ones(nb, BS, HKV, device=dev, dtype=torch.float32); vsc = torch.ones_like(ksc)
for lo in range(0, ctx, 4096):
    n = min(4096, ctx - lo)
    i4.reshape_and_cache_int4(torch.randn(n, HKV, D, device=dev, dtype=dt), torch.randn(n, HKV, D, device=dev, dtype=dt),
                              kc, vc, torch.arange(lo, lo + n, device=dev), k_scale_cache=ksc, v_scale_cache=vsc)
q = torch.randn(N * Q, HQ, D, device=dev, dtype=dt); out = torch.empty_like(q)
thr, segs = 40, 16
kw = dict(q=q, k_cache=kc, v_cache=vc, out=out, cu_seqlens_q=torch.tensor([0, Q], device=dev, dtype=torch.int32),
          max_seqlen_q=Q, seqused_k=torch.tensor([ctx], device=dev, dtype=torch.int32), max_seqlen_k=ctx,
          softmax_scale=D ** -0.5, window_size=(-1, -1), block_table=torch.arange(nb, device=dev, dtype=torch.int32).view(1, -1),
          softcap=0, sinks=None, alibi_slopes=None, use_alibi_sqrt=False, qq_bias=None, output_scale=None,
          mm_prefix_range=None, k_scale_cache=ksc, v_scale_cache=vsc, seq_threshold_3D=thr, num_par_softmax_segments=segs,
          softmax_segm_output=torch.empty(thr, HQ, segs, D, device=dev), softmax_segm_max=torch.empty(thr, HQ, segs, device=dev),
          softmax_segm_expsum=torch.empty(thr, HQ, segs, device=dev))
def medir(fn, reps=30):
    fn(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(reps): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / reps * 1e3
def llamada():
    k2 = dict(kw); q_ = k2.pop("q"); kc_ = k2.pop("k_cache"); vc_ = k2.pop("v_cache"); o_ = k2.pop("out")
    i4.unified_attention_int4(q_, kc_, vc_, o_, **k2)
g._ACTIVO = False
print(f"2D serie (sin PN124)          {medir(llamada):6.2f} ms", flush=True)
ref = out.clone()
import vllm.v1.attention.ops.triton_unified_attention as tua
orig_tile = tua._get_tile_size
g._ACTIVO = True; g._es_ampere = True
def aplanada():
    plano = g._aplanar(dict(kw, **{"q": q}))
    k2 = dict(plano); q_ = k2.pop("q"); kc_ = k2.pop("k_cache"); vc_ = k2.pop("v_cache"); o_ = k2.pop("out")
    i4.unified_attention_int4(q_, kc_, vc_, o_, **k2)
for tile in (16, 32, 64, 128):
    tua._get_tile_size = lambda hs, sw, es, is_prefill, _t=tile: 32 if is_prefill else _t
    try:
        ms = medir(aplanada)
        err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
        print(f"3D aplanado TILE_DECODE={tile:<4} {ms:6.2f} ms  err {err:.1e}", flush=True)
    except Exception as e:
        print(f"3D aplanado TILE_DECODE={tile:<4} no entra: {str(e)[:60]}", flush=True)
tua._get_tile_size = orig_tile
