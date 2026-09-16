"""PN124: el decode spec aplanado al kernel 3D debe dar la MISMA salida que el 2D."""
import os, time, torch
os.environ["GENESIS_ENABLE_PN124_TRITON_AMPERE"] = "1"
import vllm.v1.attention.ops.triton_unified_attention as tua
import vllm._genesis.triton_attn_ampere as g
dev, dt = "cuda", torch.float16
HQ, HKV, D, BS = 12, 2, 256, 832
torch.manual_seed(0)
def caso(N, Q, ctxs):
    nb = max(ctxs) // BS + 2
    k = torch.randn(nb * N, BS, HKV, D, device=dev, dtype=dt); v = torch.randn_like(k)
    q = torch.randn(N * Q, HQ, D, device=dev, dtype=dt)
    bt = torch.arange(nb * N, device=dev, dtype=torch.int32).view(N, nb)
    cu = torch.arange(0, N * Q + 1, Q, device=dev, dtype=torch.int32)
    sl = torch.tensor(ctxs, device=dev, dtype=torch.int32)
    thr, segs = 40, 16
    base = dict(q=q, k=k, v=v, cu_seqlens_q=cu, max_seqlen_q=Q, seqused_k=sl, max_seqlen_k=max(ctxs),
                softmax_scale=D ** -0.5, causal=True, alibi_slopes=None, window_size=(-1, -1), block_table=bt,
                softcap=0, sinks=None, q_descale=None, k_descale=None, v_descale=None,
                seq_threshold_3D=thr, num_par_softmax_segments=segs,
                softmax_segm_output=torch.empty(thr, HQ, segs, D, device=dev),
                softmax_segm_max=torch.empty(thr, HQ, segs, device=dev),
                softmax_segm_expsum=torch.empty(thr, HQ, segs, device=dev))
    def t(f, reps=20):
        f(); torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(reps): f()
        torch.cuda.synchronize(); return (time.perf_counter() - t0) / reps * 1e3
    o2 = torch.empty_like(q); o3 = torch.empty_like(q)
    ms2 = t(lambda: tua.unified_attention(out=o2, **base))
    g._APLANAR = True
    ms3 = t(lambda: g.llamar(tua.unified_attention, out=o3, **base))
    err = ((o2.float() - o3.float()).norm() / o2.float().norm()).item()
    print(f"N={N} Q={Q} ctx={ctxs[:3]}...: 2D {ms2:6.2f} ms  aplanado-3D {ms3:6.2f} ms  {ms2/ms3:4.1f}x  err relativo {err:.2e}", flush=True)
caso(1, 4, [57000])
caso(4, 4, [30000, 12000, 45000, 800])
caso(10, 4, [8000, 16000, 3000, 20000, 900, 12000, 5000, 7000, 11000, 2500])
caso(2, 2, [40000, 1000])
