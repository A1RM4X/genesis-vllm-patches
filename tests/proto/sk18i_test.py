"""SK-18i offline: KV INT4 de PN131 (escritura, decode entero, prefill decuantizado) sobre una
KV con la forma real de vLLM (int4_per_token_head: 264 B por token-cabeza, bloques de 832).

Uso: sk18i_test.py [N] [capa]   (por defecto 5000, capa 35)
Compara contra la atencion float de referencia y, de paso, contra el camino int8.
"""
import sys, types, time, torch
import sk18_a0bis_ptx as A
from vllm._genesis import sk18_attn as P
from vllm.v1.kv_cache_interface import KVQuantMode
dev = "cuda"; D = 256; NH = 2; G = 6; BS = 832
torch.manual_seed(0)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
CAPA = int(sys.argv[2]) if len(sys.argv) > 2 else 35
MODOS = {"int4": (KVQuantMode.INT4_PER_TOKEN_HEAD, 264), "int8": (KVQuantMode.INT8_PER_TOKEN_HEAD, 520)}


def meta(qsl, seq, bt, nact):
    from vllm.v1.attention.backends.triton_attn import MIN_LAUNCH_GRID_SIZE_2D, NUM_PAR_SOFTMAX_SEGMENTS
    thr = MIN_LAUNCH_GRID_SIZE_2D // NH
    return types.SimpleNamespace(
        num_actual_tokens=nact, max_query_len=int((qsl[1:] - qsl[:-1]).max()), query_start_loc=qsl.to(dev),
        max_seq_len=int(seq.max()), seq_lens=seq.to(dev), block_table=bt.to(dev), causal=True,
        genesis_qsl_cpu=qsl.cpu(), seq_threshold_3D=thr, num_par_softmax_segments=NUM_PAR_SOFTMAX_SEGMENTS,
        softmax_segm_output=torch.empty(thr, 12, NUM_PAR_SOFTMAX_SEGMENTS, 256, device=dev),
        softmax_segm_max=torch.empty(thr, 12, NUM_PAR_SOFTMAX_SEGMENTS, device=dev),
        softmax_segm_expsum=torch.empty(thr, 12, NUM_PAR_SOFTMAX_SEGMENTS, device=dev),
        mm_prefix_range_tensor=None, rswa_prefix_lens=None, rswa_window=None)


def medir(f, n=20):
    for _ in range(3): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e3


q, k, v = A.cargar(CAPA); Nt = k.shape[0]
reps = (N + Nt - 1) // Nt
kk = k.repeat(reps, 1, 1)[:N].contiguous(); vv = v.repeat(reps, 1, 1)[:N].contiguous()
qq = q.repeat(reps, 1, 1)[:N].contiguous()
pos = torch.arange(N - 4, N, device=dev)
ref = A.referencia(qq, kk, vv, pos).float()

nblocks = (N + BS - 1) // BS + 6
perm = torch.randperm(nblocks)[: (N + BS - 1) // BS].to(torch.int32)
slots = (perm[torch.arange(N) // BS].to(torch.int64) * BS + torch.arange(N) % BS).to(dev)
bt = perm[None, :].clone()
qsl = torch.tensor([0, 4], dtype=torch.int32)

for nombre, (qm, cont) in MODOS.items():
    impl = types.SimpleNamespace(_kv_quant_mode=qm, head_size=D, num_heads=12,
                                 alibi_slopes=None, sinks=None, sliding_window=(-1, -1), logits_soft_cap=0,
                                 scale=1.0 / 16, chunk_lookback=-1, use_td=False, num_kv_heads=NH)
    layer = types.SimpleNamespace(_k_scale=torch.ones(1, device=dev), _v_scale=torch.ones(1, device=dev),
                                  layer_name=f"capa{CAPA}")
    kv = torch.zeros(nblocks, BS, NH, cont, dtype=torch.uint8 if nombre == "int4" else torch.int8,
                     device=dev).permute(0, 2, 1, 3)
    t_esc = 0.0
    for t0 in range(0, N, 7488):
        t1 = min(N, t0 + 7488)
        # metadata como en el servidor: una secuencia, seq_lens = tokens escritos hasta aca
        P._md_forzada = meta(torch.tensor([0, t1 - t0], dtype=torch.int32),
                             torch.tensor([t1], dtype=torch.int32), bt, t1 - t0)
        P.escribir(impl, layer, kk[t0:t1].half(), vv[t0:t1].half(), kv, slots[t0:t1])
    P._md_forzada = None
    t_esc = medir(lambda: P.escribir(impl, layer, kk[:min(N, 7488)].half(), vv[:min(N, 7488)].half(), kv,
                                     slots[:min(N, 7488)]), n=5)
    md = meta(qsl, torch.tensor([N], dtype=torch.int32), bt, 4)
    out = torch.zeros(4, 12, D, dtype=torch.float16, device=dev)
    P._decode_uniforme(impl, qq[pos].half(), kv, md, out, 1, 4, False)
    o1 = out.clone()
    iguales = True
    for _ in range(3):
        out.zero_()
        P._decode_uniforme(impl, qq[pos].half(), kv, md, out, 1, 4, False)
        iguales &= bool((out == o1).all())
    err_d = ((o1.float() - ref).norm(dim=-1) / ref.norm(dim=-1)).mean().item()
    # prefill: decuantiza y corre el camino de vLLM
    out2 = torch.zeros_like(out)
    import os
    os.environ["GENESIS_PN131_PREFILL"] = "flashinfer"
    P._prefill_decuant(impl, layer, qq[pos].half(), kv, md, out2)
    err_p = ((out2.float() - ref).norm(dim=-1) / ref.norm(dim=-1)).mean().item()
    t_dec = medir(lambda: P._decode_uniforme(impl, qq[pos].half(), kv, md, out, 1, 4, False))
    print(f"{nombre}: N={N} capa={CAPA} | decode err {100*err_d:.3f}% | prefill(decuant) err {100*err_p:.3f}% "
          f"| deterministico={iguales} | decode {t_dec:.3f} ms | escritura 7488 tok {t_esc:.3f} ms", flush=True)

# ── lote de 3 secuencias de largos distintos (paridad de slots, colas impares) ──
print("\nlote de 3:")
largos = [min(N, 7000), min(N, 3001), 833]
for nombre, (qm, cont) in MODOS.items():
    impl = types.SimpleNamespace(_kv_quant_mode=qm, head_size=D, num_heads=12,
                                 alibi_slopes=None, sinks=None, sliding_window=(-1, -1), logits_soft_cap=0,
                                 scale=1.0 / 16, chunk_lookback=-1, use_td=False, num_kv_heads=NH)
    layer = types.SimpleNamespace(_k_scale=torch.ones(1, device=dev), _v_scale=torch.ones(1, device=dev),
                                  layer_name=f"capa{CAPA}")
    nb2 = sum((L + BS - 1) // BS for L in largos) + 4
    kv = torch.zeros(nb2, BS, NH, cont, dtype=torch.uint8 if nombre == "int4" else torch.int8,
                     device=dev).permute(0, 2, 1, 3)
    maxp = max((L + BS - 1) // BS for L in largos)
    bt2 = torch.zeros(len(largos), maxp, dtype=torch.int32)
    libre = torch.randperm(nb2).tolist()
    outs, refs = [], []
    for i, L in enumerate(largos):
        np_ = (L + BS - 1) // BS
        pg = torch.tensor(libre[:np_], dtype=torch.int32); libre = libre[np_:]
        bt2[i, :np_] = pg
        sl = (pg[torch.arange(L) // BS].to(torch.int64) * BS + torch.arange(L) % BS).to(dev)
        for t0 in range(0, L, 7488):
            t1 = min(L, t0 + 7488)
            P._md_forzada = meta(torch.tensor([0, t1 - t0], dtype=torch.int32),
                                 torch.tensor([t1], dtype=torch.int32), bt2[i:i + 1], t1 - t0)
            P._md_forzada.seq_lens_b = i
            P.escribir(impl, layer, kk[t0:t1].half(), vv[t0:t1].half(), kv, sl[t0:t1])
        P._md_forzada = None
        refs.append(A.referencia(qq, kk, vv, torch.arange(L - 4, L, device=dev)).float())
    seq = torch.tensor(largos, dtype=torch.int32)
    qsl3 = torch.tensor([0, 4, 8, 12], dtype=torch.int32)
    md3 = meta(qsl3, seq, bt2, 12)
    qcat = torch.cat([qq[L - 4:L] for L in largos]).half()
    out3 = torch.zeros(12, 12, D, dtype=torch.float16, device=dev)
    P._decode_uniforme(impl, qcat, kv, md3, out3, 3, 4, False)
    errs = [((out3[4 * i:4 * i + 4].float() - refs[i]).norm(dim=-1) / refs[i].norm(dim=-1)).mean().item()
            for i in range(3)]
    print(f"  {nombre}: largos={largos} err={[round(100*e, 3) for e in errs]} %")
