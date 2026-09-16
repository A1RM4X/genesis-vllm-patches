"""Prototipo: rollback del MTP en GDN con CINTA en vez de K copias del estado.

Upstream (spec decode): lee el estado de la columna num_accepted-1 y escribe el
estado completo tras CADA uno de los K+1 tokens (K+1 escrituras de 786 KB por
capa por request) -> reserva K bloques extra del pool por grupo.

Cinta: un solo slot. Tras el paso guarda el estado tras el token 0 (siempre
aceptado) y las ENTRADAS (k, v, a, b) de los tokens 1..K. Al paso siguiente
reproduce num_accepted-1 entradas sobre el slot y sigue.
"""
import sys, time, torch
from vllm.triton_utils import tl, triton
from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update as ref_update)


@triton.jit
def _paso(b_h, b_k, b_v, a_raw, b_raw, A_log, dt_bias, beta, threshold, IS_L2: tl.constexpr):
    x = a_raw + dt_bias
    sp = tl.where(beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x)
    g = -tl.exp(A_log) * sp
    bb = tl.sigmoid(b_raw)
    if IS_L2:
        b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
    b_h = b_h * tl.exp(g)
    b_v = (b_v - tl.sum(b_h * b_k[None, :], 1)) * bb
    b_h = b_h + b_v[:, None] * b_k[None, :]
    return b_h, b_k


@triton.jit(do_not_specialize=["N"])
def _kernel_cinta(A_log, a, b, dt_bias, beta, threshold, q, k, v, o, h, cu, sidx, nrep,
                  ck, cv, ca, cb, scale, N,
                  H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                  BK: tl.constexpr, BV: tl.constexpr, TM: tl.constexpr, IS_L2: tl.constexpr):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos = tl.load(cu + i_n).to(tl.int64); eos = tl.load(cu + i_n + 1).to(tl.int64)
    T = eos - bos
    if T == 0:
        return
    s = tl.load(sidx + i_n).to(tl.int64)
    if s <= 0:
        return
    o_k = i_k * BK + tl.arange(0, BK); o_v = i_v * BV + tl.arange(0, BV)
    mk = o_k < K; mv = o_v < V; mh = mv[:, None] & mk[None, :]
    Al = tl.load(A_log + i_hv).to(tl.float32); db = tl.load(dt_bias + i_hv).to(tl.float32)
    p_h = h + s * HV * V * K + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(p_h, mask=mh, other=0).to(tl.float32)
    # 1) reproducir la cinta del paso anterior
    r = tl.load(nrep + i_n).to(tl.int64)
    for j in range(0, r):
        base = s * TM + j
        rk = tl.load(ck + (base * H + i_h) * K + o_k, mask=mk, other=0).to(tl.float32)
        rv = tl.load(cv + (base * HV + i_hv) * V + o_v, mask=mv, other=0).to(tl.float32)
        ra = tl.load(ca + base * HV + i_hv).to(tl.float32)
        rb = tl.load(cb + base * HV + i_hv).to(tl.float32)
        b_h, rk = _paso(b_h, rk, rv, ra, rb, Al, db, beta, threshold, IS_L2)
    # 2) tokens del paso actual
    p_q = q + (bos * H + i_h) * K + o_k; p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v; p_a = a + bos * HV + i_hv; p_b = b + bos * HV + i_hv
    p_o = o + (bos * HV + i_hv) * V + o_v
    for t in range(0, T):
        bq = tl.load(p_q, mask=mk, other=0).to(tl.float32)
        bk = tl.load(p_k, mask=mk, other=0)
        bv = tl.load(p_v, mask=mv, other=0)
        ba = tl.load(p_a); bb_ = tl.load(p_b)
        b_h, bk2 = _paso(b_h, bk.to(tl.float32), bv.to(tl.float32), ba.to(tl.float32),
                         bb_.to(tl.float32), Al, db, beta, threshold, IS_L2)
        if IS_L2:
            bq = bq * tl.rsqrt(tl.sum(bq * bq) + 1e-6)
        bq = bq * scale
        tl.store(p_o, tl.sum(b_h * bq[None, :], 1).to(p_o.dtype.element_ty), mask=mv)
        if t == 0:  # la unica escritura del estado completo
            tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mh)
        p_q += H * K; p_k += H * K; p_v += HV * V; p_a += HV; p_b += HV; p_o += HV * V


@triton.jit
def _kernel_escribir_cinta(k, v, a, b, ck, cv, ca, cb, sidx,
                           T: tl.constexpr, TM: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
                           K: tl.constexpr, V: tl.constexpr, BHK: tl.constexpr, BHV: tl.constexpr,
                           BH: tl.constexpr):
    """Copia k, v, a, b de los tokens 1..TM de la request i_n a su cinta. Lanzamiento
    aparte del kernel de recurrencia: adentro habia carrera lectura/escritura."""
    i_n = tl.program_id(0)
    s = tl.load(sidx + i_n).to(tl.int64)
    if s <= 0:
        return
    ohk = tl.arange(0, BHK); ohv = tl.arange(0, BHV); oh = tl.arange(0, BH)
    mhk = ohk < H * K; mhv = ohv < HV * V; mh = oh < HV
    for t in range(0, TM):
        src = i_n * T + t + 1
        dst = s * TM + t
        tl.store(ck + dst * H * K + ohk, tl.load(k + src * H * K + ohk, mask=mhk), mask=mhk)
        tl.store(cv + dst * HV * V + ohv, tl.load(v + src * HV * V + ohv, mask=mhv), mask=mhv)
        tl.store(ca + dst * HV + oh, tl.load(a + src * HV + oh, mask=mh), mask=mh)
        tl.store(cb + dst * HV + oh, tl.load(b + src * HV + oh, mask=mh), mask=mh)


def cinta_update(A_log, a, b, dt_bias, q, k, v, h, cu, sidx, nrep, cinta):
    _, T, H, K = k.shape; HV, V = v.shape[2], v.shape[3]
    N = len(cu) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    o = q.new_empty(v.shape[1:])
    ck, cv, ca, cb = cinta
    _kernel_cinta[(1, triton.cdiv(V, BV), N * HV)](
        A_log, a.contiguous(), b.contiguous(), dt_bias, 1.0, 20.0, q.contiguous(),
        k.contiguous(), v.contiguous(), o, h, cu, sidx, nrep, ck, cv, ca, cb,
        K ** -0.5, N, H=H, HV=HV, K=K, V=V, BK=BK, BV=BV, TM=ck.shape[1], IS_L2=True,
        num_warps=4, num_stages=3)
    # La cinta se escribe DESPUES, en un lanzamiento aparte (ver el kernel).
    TM = ck.shape[1]
    _kernel_escribir_cinta[(N,)](k, v, a, b, ck, cv, ca, cb, sidx, T=T // N, TM=TM, H=H, HV=HV,
                                 K=K, V=V, BHK=triton.next_power_of_2(H * K),
                                 BHV=triton.next_power_of_2(HV * V), BH=triton.next_power_of_2(HV))
    return o


def main():
    torch.manual_seed(0)
    dev, dt = "cuda", torch.float16
    H, HV, K, V, SPEC = 8, 24, 128, 128, 3           # por rank con TP=2
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    T = SPEC + 1
    S = 1 + N * (SPEC + 1)                            # slot 0 = null
    A_log = torch.randn(HV, device=dev, dtype=torch.float32) * 0.5
    dt_bias = torch.randn(HV, device=dev, dtype=torch.float32) * 0.5
    h_ref = torch.randn(S, HV, V, K, device=dev, dtype=dt) * 0.05
    h_cin = h_ref.clone()
    cols = torch.arange(1, S, device=dev, dtype=torch.int32).view(N, SPEC + 1)
    sidx_cin = cols[:, 0].contiguous()
    cinta = (torch.zeros(S, SPEC, H, K, device=dev, dtype=dt), torch.zeros(S, SPEC, HV, V, device=dev, dtype=dt),
             torch.zeros(S, SPEC, HV, device=dev, dtype=dt), torch.zeros(S, SPEC, HV, device=dev, dtype=dt))
    cu = torch.arange(0, (N + 1) * T, T, device=dev, dtype=torch.int32)
    acc = torch.ones(N, device=dev, dtype=torch.int32)
    peor = 0.0
    for paso in range(40):
        q = torch.randn(1, N * T, H, K, device=dev, dtype=dt)
        k = torch.randn(1, N * T, H, K, device=dev, dtype=dt)
        v = torch.randn(1, N * T, HV, V, device=dev, dtype=dt)
        a = torch.randn(N * T, HV, device=dev, dtype=dt)
        b = torch.randn(N * T, HV, device=dev, dtype=dt)
        o_ref, _ = ref_update(A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=q, k=k, v=v,
                              initial_state=h_ref, inplace_final_state=True, cu_seqlens=cu,
                              ssm_state_indices=cols, num_accepted_tokens=acc,
                              use_qk_l2norm_in_kernel=True)
        o_cin = cinta_update(A_log, a, b, dt_bias, q, k, v, h_cin, cu, sidx_cin, acc - 1, cinta)
        err = ((o_ref.float() - o_cin.float()).norm() / o_ref.float().norm()).item()
        peor = max(peor, err)
        acc = torch.randint(1, SPEC + 2, (N,), device=dev, dtype=torch.int32)
    print(f"N={N}: error relativo maximo de la salida en 40 pasos = {peor:.2e}")

    # velocidad: un paso de decode spec, estado ya caliente
    def bench(f, reps=2000):
        # Capturado en CUDA graph, como corre el decode en produccion: mide GPU,
        # no el overhead de lanzar kernels desde Python.
        st = torch.cuda.Stream()
        with torch.cuda.stream(st):
            for _ in range(5): f()
        torch.cuda.current_stream().wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): f()
        torch.cuda.synchronize(); [g.replay() for _ in range(50)]; torch.cuda.synchronize()
        t0 = time.perf_counter(); [g.replay() for _ in range(reps)]; torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1e3
    acc = torch.full((N,), 3, device=dev, dtype=torch.int32)  # TAR tipico ~ 2 aceptados
    f_ref = lambda: ref_update(A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=q, k=k, v=v, initial_state=h_ref,
                               inplace_final_state=True, cu_seqlens=cu, ssm_state_indices=cols,
                               num_accepted_tokens=acc, use_qk_l2norm_in_kernel=True)
    f_cin = lambda: cinta_update(A_log, a, b, dt_bias, q, k, v, h_cin, cu, sidx_cin, acc - 1, cinta)
    r, c = bench(f_ref), bench(f_cin)
    print(f"N={N}: upstream {r:.3f} ms/capa   cinta (2 replays) {c:.3f} ms/capa   {r/c:.2f}x")


if __name__ == "__main__":
    main()
