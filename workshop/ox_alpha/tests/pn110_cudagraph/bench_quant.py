#!/usr/bin/env python3
"""bench_quant.py — por que el quant de activacion cuesta 2.2us a M=4.

El kernel actual lanza grid=(M,): una fila por programa. En decode M=4, o sea
**4 CTAs sobre 82 SMs**. Mueve 40 KB. A 936 GB/s eso son 0.04us, asi que los
2.2us medidos son latencia, no ancho de banda.

Prueba variantes con mas paralelismo por fila.
"""
import statistics, sys, torch, triton, triton.language as tl

@triton.jit
def _actual(x_ptr, q_ptr, s_ptr, K, sx, sq, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK); mask = offs < K
    x = tl.load(x_ptr + row*sx + offs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-30)
    xq = x * (127.0/amax)
    q = (xq + tl.where(xq >= 0.0, 0.5, -0.5)).to(tl.int32)
    tl.store(q_ptr + row*sq + offs, tl.minimum(tl.maximum(q,-127),127).to(tl.int8), mask=mask)
    tl.store(s_ptr + row, amax*(1.0/127.0))

@triton.jit
def _2d(x_ptr, q_ptr, s_ptr, K, sx, sq, BLOCK: tl.constexpr, NSPLIT: tl.constexpr):
    """Una fila por program_id(0), K partido en NSPLIT por program_id(1).
    Cada programa recorre su franja dos veces (amax y quant) pero con NSPLIT
    veces mas CTAs. amax se re-reduce localmente: exacto sin atomics porque cada
    programa necesita el amax de TODA la fila -> se recalcula leyendo la fila
    entera pero escribiendo solo su franja."""
    row = tl.program_id(0); part = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    amax = tl.zeros((), dtype=tl.float32) + 1e-30
    for k0 in range(0, K, BLOCK):
        v = tl.load(x_ptr + row*sx + k0 + offs, mask=k0+offs < K, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(v), axis=0))
    base = part * BLOCK
    v = tl.load(x_ptr + row*sx + base + offs, mask=base+offs < K, other=0.0).to(tl.float32)
    xq = v * (127.0/amax)
    q = (xq + tl.where(xq >= 0.0, 0.5, -0.5)).to(tl.int32)
    tl.store(q_ptr + row*sq + base + offs, tl.minimum(tl.maximum(q,-127),127).to(tl.int8), mask=base+offs < K)
    tl.store(s_ptr + row, amax*(1.0/127.0))

def bench(fn, *a, reps=48):
    """Mide DENTRO de un CUDA graph. En eager el lanzamiento cuesta ~11us y
    tapa por completo un kernel de 2us: medir asi da 11.26us para todo y las
    variantes parecen identicas. Es la misma trampa que ya hizo dar 21% donde
    en el server real habia 4%."""
    for _ in range(3): fn(*a)          # calentar / compilar
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps): fn(*a)
    torch.cuda.synchronize()
    e0,e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ts=[]
    for _ in range(20):
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1)*1000/reps)
    return statistics.median(ts)

dev=torch.device("cuda")
M=int(sys.argv[1]) if len(sys.argv)>1 else 4
print(f"M={M}\n{'K':>7}{'actual us':>11}{'CTAs':>6}{'  |':>3}{'mejor variante':>28}{'us':>8}{'CTAs':>7}{'gana':>8}")
for K in (3072, 5120, 8704):
    x=torch.randn(M,K,dtype=torch.bfloat16,device=dev)
    q=torch.empty(M,K,dtype=torch.int8,device=dev); s=torch.empty(M,dtype=torch.float32,device=dev)
    B=triton.next_power_of_2(K)
    t0=bench(lambda: _actual[(M,)](x,q,s,K,x.stride(0),q.stride(0),BLOCK=B,num_warps=8,num_stages=1))
    mejor=(None,1e9,0)
    for nsplit,warps in ((4,4),(8,4),(8,2),(16,2)):
        blk=triton.next_power_of_2(-(-K//nsplit))
        real=-(-K//blk)
        try:
            t=bench(lambda: _2d[(M,real)](x,q,s,K,x.stride(0),q.stride(0),BLOCK=blk,NSPLIT=real,num_warps=warps))
            if t<mejor[1]: mejor=(f"BLOCK={blk} w={warps}", t, M*real)
        except Exception as e:
            pass
    print(f"{K:>7}{t0:>11.2f}{M:>6}{'  |':>3}{mejor[0] or '-':>28}{mejor[1]:>8.2f}{mejor[2]:>7}{t0/mejor[1]:>7.2f}x")
