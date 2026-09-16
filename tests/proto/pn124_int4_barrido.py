"""Barrido de parametros del kernel INT4 per-token-head (prefill) con formas del Qwen3.8."""
import os, time, torch
os.environ["GENESIS_ENABLE_PN124_TRITON_AMPERE"] = "1"
import vllm._genesis.triton_attn_ampere as g
from vllm.v1.attention.ops.int4_per_token_head import reshape_and_cache_int4, unified_attention_int4
dev = "cuda"; dt = torch.float16
HQ, HKV, D, BS = 12, 2, 256, 1616
torch.manual_seed(0)
ctx, nq = 14976, 6464
nb = ctx // BS + 2
kc = torch.zeros(nb, BS, HKV, D // 2, device=dev, dtype=torch.uint8)
vc = torch.zeros_like(kc)
ksc = torch.ones(nb, BS, HKV, device=dev, dtype=torch.float32); vsc = torch.ones_like(ksc)
for lo in range(0, ctx, 4096):
    n = min(4096, ctx - lo)
    reshape_and_cache_int4(torch.randn(n, HKV, D, device=dev, dtype=dt), torch.randn(n, HKV, D, device=dev, dtype=dt),
                           kc, vc, torch.arange(lo, lo + n, device=dev, dtype=torch.int64),
                           k_scale_cache=ksc, v_scale_cache=vsc)
q = torch.randn(nq, HQ, D, device=dev, dtype=dt); out = torch.empty_like(q)
bt = torch.arange(nb, device=dev, dtype=torch.int32).view(1, -1)
cu = torch.tensor([0, nq], device=dev, dtype=torch.int32); sl = torch.tensor([ctx], device=dev, dtype=torch.int32)
def f():
    unified_attention_int4(q, kc, vc, out, cu_seqlens_q=cu, max_seqlen_q=nq, seqused_k=sl, max_seqlen_k=ctx,
        softmax_scale=D ** -0.5, window_size=(-1, -1), block_table=bt, softcap=0, sinks=None, alibi_slopes=None,
        use_alibi_sqrt=False, qq_bias=None, output_scale=None, mm_prefix_range=None,
        k_scale_cache=ksc, v_scale_cache=vsc)
def medir(reps=3):
    f(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(reps): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / reps * 1e3
g._ACTIVO = False
base = medir(); ref = out.clone()
print(f"generico                          {base:7.1f} ms", flush=True)
g._ACTIVO = True; g._es_ampere = True
for bm in (16, 32, 64):
    for tile in (16, 32, 64):
        for st in (1, 2, 3):
            for w in (4, 8):
                g._CONFIGS[0] = (bm, tile, st, w); g._nivel = 0
                try:
                    ms = medir()
                    err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
                    print(f"BM={bm:<3} TILE={tile:<3} st={st} w={w}   {ms:7.1f} ms  {base/ms:4.2f}x  err {err:.1e}", flush=True)
                except Exception as e:
                    print(f"BM={bm:<3} TILE={tile:<3} st={st} w={w}   no entra: {str(e)[:40]}", flush=True)
                torch.cuda.empty_cache()
