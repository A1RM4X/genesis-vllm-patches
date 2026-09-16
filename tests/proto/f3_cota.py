"""F3 — cota: que fraccion de la atencion de prefill es el producto Q.K.

Forma real por placa (TP=2): chunk de M queries, 12 cabezas Q, 2 KV, head 256,
contexto L (causal alineado abajo a la derecha).
  atencion completa : vllm_flash_attn flash_attn_varlen_func (fp16)
  solo Q.K fp16     : GEMM cuBLAS de q x k (mismos flops que el producto punto)
  solo Q.K int8     : cutlass_scaled_mm int8 x int8 (lo que haria un Q.K INT8 ideal)
"""
import sys, time, torch
from vllm.vllm_flash_attn import flash_attn_varlen_func
from vllm import _custom_ops as ops
dev = "cuda"; torch.manual_seed(0)
Hq, Hk, D = 12, 2, 256
M = 7488
def cron(f, it=3):
    for _ in range(2): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / it
for L in [int(x) for x in (sys.argv[1:] or ["16000", "64000"])]:
    q = torch.randn(M, Hq, D, device=dev, dtype=torch.float16)
    k = torch.randn(L, Hk, D, device=dev, dtype=torch.float16)
    v = torch.randn(L, Hk, D, device=dev, dtype=torch.float16)
    cq = torch.tensor([0, M], device=dev, dtype=torch.int32); ck = torch.tensor([0, L], device=dev, dtype=torch.int32)
    t_fa = cron(lambda: flash_attn_varlen_func(q, k, v, cu_seqlens_q=cq, cu_seqlens_k=ck, max_seqlen_q=M, max_seqlen_k=L, causal=True))
    # Q.K: por cabeza KV, 6 cabezas Q contra L keys; en tandas de 1024 queries (memoria)
    qg = q.view(M, Hk, Hq // Hk, D)
    def qk_f16():
        for h in range(Hk):
            kk = k[:, h, :]
            for lo in range(0, M, 1024):
                a = qg[lo:lo + 1024, h].reshape(-1, D)
                _ = a @ kk.t()
    t_qk16 = cron(qk_f16, it=1)
    q8, qs, _ = ops.scaled_int8_quant(q.reshape(-1, D).contiguous(), symmetric=True)
    k8, ks, _ = ops.scaled_int8_quant(k.reshape(-1, D).contiguous(), symmetric=True)
    q8 = q8.view(M, Hk, Hq // Hk, D); qs = qs.view(M, Hk, Hq // Hk, 1); k8 = k8.view(L, Hk, D); ks = ks.view(L, Hk)
    def qk_i8():
        for h in range(Hk):
            kk = k8[:, h, :].t(); kss = ks[:, h].reshape(1, -1).contiguous()
            for lo in range(0, M, 1024):
                a = q8[lo:lo + 1024, h].reshape(-1, D).contiguous()
                _ = ops.cutlass_scaled_mm(a, kk, qs[lo:lo + 1024, h].reshape(-1, 1).contiguous(), kss, torch.float16)
    t_qk8 = cron(qk_i8, it=1)
    print(f"L={L:6d}: atencion FA {t_fa*1e3:8.1f} ms | Q.K fp16 {t_qk16*1e3:8.1f} ms ({100*t_qk16/t_fa:4.0f}% de FA) | "
          f"Q.K int8 {t_qk8*1e3:8.1f} ms ({t_qk16/t_qk8:.2f}x vs fp16)", flush=True)
    del q, k, v, q8, k8; torch.cuda.empty_cache()
