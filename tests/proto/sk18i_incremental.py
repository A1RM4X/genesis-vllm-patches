"""SK-18i: decode INCREMENTAL como en el servidor (escribe 5 tokens por paso y decodifica),
comparando el camino con espejo int8 de la ventana contra el mismo sin espejo y contra float.

Sirve para cazar el problema del espejo en el servidor: el test de una sola escritura no lo ve.
Uso: sk18i_incremental.py [N_base] [pasos]
"""
import sys, types, torch
import sk18_a0bis_ptx as A
from vllm._genesis import sk18_attn as P
from vllm.v1.kv_cache_interface import KVQuantMode
dev = "cuda"; D = 256; NH = 2; G = 6; BS = 832
torch.manual_seed(0)
N0 = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
PASOS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
L = 5                                   # tokens por paso (MTP K=4)


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


q, k, v = A.cargar(35); Nt = k.shape[0]
NMAX = N0 + PASOS * L + 16
reps = (NMAX + Nt - 1) // Nt
kk = k.repeat(reps, 1, 1)[:NMAX].contiguous(); vv = v.repeat(reps, 1, 1)[:NMAX].contiguous()
qq = q.repeat(reps, 1, 1)[:NMAX].contiguous()
nblocks = (NMAX + BS - 1) // BS + 6
perm = torch.randperm(nblocks)[: (NMAX + BS - 1) // BS].to(torch.int32)
slots = (perm[torch.arange(NMAX) // BS].to(torch.int64) * BS + torch.arange(NMAX) % BS).to(dev)
bt = perm[None, :].clone()

impl = types.SimpleNamespace(_kv_quant_mode=KVQuantMode.INT4_PER_TOKEN_HEAD, head_size=D, num_heads=12,
                             alibi_slopes=None, sinks=None, sliding_window=(-1, -1), logits_soft_cap=0,
                             scale=1.0 / 16, chunk_lookback=-1, use_td=False, num_kv_heads=NH)
layer = types.SimpleNamespace(_k_scale=torch.ones(1, device=dev), _v_scale=torch.ones(1, device=dev),
                              layer_name="capa35")
kv = torch.zeros(nblocks, BS, NH, 264, dtype=torch.uint8, device=dev).permute(0, 2, 1, 3)
# prefill
P._md_forzada = meta(torch.tensor([0, N0], dtype=torch.int32), torch.tensor([N0], dtype=torch.int32), bt, N0)
P.escribir(impl, layer, kk[:N0].half(), vv[:N0].half(), kv, slots[:N0])

out = torch.zeros(L, 12, D, dtype=torch.float16, device=dev)
peor = 0.0
for paso in range(PASOS):
    n = N0 + paso * L
    P._md_forzada = meta(torch.tensor([0, L], dtype=torch.int32), torch.tensor([n + L], dtype=torch.int32), bt, L)
    P.escribir(impl, layer, kk[n:n + L].half(), vv[n:n + L].half(), kv, slots[n:n + L])
    P._md_forzada = None
    N = n + L
    md = meta(torch.tensor([0, L], dtype=torch.int32), torch.tensor([N], dtype=torch.int32), bt, L)
    pos = torch.arange(N - L, N, device=dev)
    ref = A.referencia(qq[:N], kk[:N], vv[:N], pos).float()
    P._decode_uniforme(impl, qq[pos].half(), kv, md, out, 1, L, False)
    err = ((out.float() - ref).norm(dim=-1) / ref.norm(dim=-1)).mean().item()
    peor = max(peor, err)
    pag = (N - 1) // BS
    if paso < 3 or err > 0.09 or paso == PASOS - 1:
        d = P._bufs[0].dueno[:4].tolist() if P.VENT else []
        print(f"paso {paso:3d} N={N} pagina={pag} err={100*err:.3f}% dueno={d} "
              f"bloques ultimas paginas={bt[0, max(0,pag-1):pag+1].tolist()}", flush=True)
print(f"VENT={P.VENT}: peor error {100*peor:.3f}%")
