"""Layout del peso: [K,N] disperso contra bloqueado por columna.

Hoy `b_col` es [K,N] con strides (1,K): contiguo en K. Un CTA con tile
[BLOCK_K=128, BLOCK_N=128] lee, por iteracion, 128 columnas de 128 bytes cada
una, y columnas consecutivas estan K bytes aparte (8704 para SK-06). O sea 128
rafagas de 128 B dispersas por 1.1 MB de VRAM. Con 40 CTAs en vuelo son ~5120
streams sequenciales concurrentes peleando por los row buffers de la GDDR6X.

Layout bloqueado: reordenar el peso a [N/128, K, 128] en la CARGA. El tile
entero de un CTA pasa a ser UNA tirada contigua de 16 KB, y el CTA completo lee
un unico bloque contiguo de K*128 = 1.1 MB. 40 streams en vez de 5120.

Es un reordenamiento OFFLINE: cuesta cero en runtime.
"""
import torch, triton, triton.language as tl


@triton.jit
def _gemm_disperso(a_ptr, b_ptr, out_ptr, a_scale_ptr, blk_ptr, M, N, K,
                   stride_am, stride_ak, stride_bk, stride_bn,
                   stride_out_m, stride_out_n, stride_blk_k, stride_blk_n,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, K // BLOCK_K):
        d = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), out_dtype=tl.float32)
        acc += d * tl.load(blk_ptr + kb * stride_blk_k + pid_n * stride_blk_n).to(tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    acc = acc * tl.load(a_scale_ptr + offs_m).to(tl.float32)[:, None]
    tl.store(out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n,
             acc.to(out_ptr.dtype.element_ty))


@triton.jit
def _gemm_bloqueado(a_ptr, b_ptr, out_ptr, a_scale_ptr, blk_ptr, M, N, K,
                    stride_am, stride_ak,
                    stride_out_m, stride_out_n, stride_blk_k, stride_blk_n,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    BLOCK_K: tl.constexpr):
    """b es [N/BLOCK_N, K, BLOCK_N] contiguo: el CTA lee una tirada sola."""
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # base del bloque de columnas + k*BLOCK_N + n  -> todo contiguo
    b_ptrs = b_ptr + pid_n * K * BLOCK_N + offs_k[:, None] * BLOCK_N + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, K // BLOCK_K):
        d = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), out_dtype=tl.float32)
        acc += d * tl.load(blk_ptr + kb * stride_blk_k + pid_n * stride_blk_n).to(tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * BLOCK_N
    acc = acc * tl.load(a_scale_ptr + offs_m).to(tl.float32)[:, None]
    out_n = pid_n * BLOCK_N + offs_n
    tl.store(out_ptr + offs_m[:, None] * stride_out_m + out_n[None, :] * stride_out_n,
             acc.to(out_ptr.dtype.element_ty))


FORMAS = [("SK-02/04", 3072, 5120), ("SK-03", 5120, 7168),
          ("SK-01", 5120, 8192), ("SK-06", 8704, 5120), ("SK-05", 5120, 17408)]
BM, BN, BK, W, S = 16, 128, 128, 8, 3

print(f"{'forma':<9}{'K':>6}{'N':>7}{'M':>3}{'disperso':>10}{'GB/s':>7}"
      f"{'bloqueado':>11}{'GB/s':>7}{'gana':>7}")
for nombre, K, N in FORMAS:
    for M in (16,):
        b = torch.randint(-127, 127, (K, N), dtype=torch.int8, device="cuda").t().contiguous().t()
        # mismo peso, reordenado a [N/BN, K, BN]
        bb = b.reshape(K, N // BN, BN).permute(1, 0, 2).contiguous()
        a = torch.randint(-127, 127, (M, K), dtype=torch.int8, device="cuda")
        o1 = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
        o2 = torch.empty_like(o1)
        asc = torch.rand(M, dtype=torch.float32, device="cuda")
        blk = torch.rand((K // BK, N // BN), dtype=torch.float32, device="cuda")
        g = (N // BN,)

        def d():
            _gemm_disperso[g](a, b, o1, asc, blk, M, N, K, a.stride(0), a.stride(1),
                              b.stride(0), b.stride(1), o1.stride(0), o1.stride(1),
                              blk.stride(0), blk.stride(1),
                              BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, num_warps=W, num_stages=S)

        def q():
            _gemm_bloqueado[g](a, bb, o2, asc, blk, M, N, K, a.stride(0), a.stride(1),
                               o2.stride(0), o2.stride(1), blk.stride(0), blk.stride(1),
                               BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, num_warps=W, num_stages=S)

        for _ in range(5):
            d(); q()
        torch.cuda.synchronize()
        dif = (o1.float() - o2.float()).abs().max().item()
        rel = dif / max(o1.float().abs().max().item(), 1e-9)
        assert rel < 1e-2, f"los dos layouts tienen que dar lo MISMO (rel={rel:.3g})"
        R = 50
        e1 = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(R)]
        e2 = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(R)]
        for r in range(R):
            e1[r][0].record(); d(); e1[r][1].record()
            e2[r][0].record(); q(); e2[r][1].record()
        torch.cuda.synchronize()
        t1 = sorted(x.elapsed_time(y) * 1000 for x, y in e1)[R // 2]
        t2 = sorted(x.elapsed_time(y) * 1000 for x, y in e2)[R // 2]
        print(f"{nombre:<9}{K:>6}{N:>7}{M:>3}{t1:9.1f}us{K*N/t1/1e3:7.0f}"
              f"{t2:10.1f}us{K*N/t2/1e3:7.0f}{t1/t2:7.2f}x")
        del b, bb, a, o1, o2
        torch.cuda.empty_cache()
