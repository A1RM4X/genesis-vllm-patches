"""PN131 offline: escritura + decode SK-18h + prefill decuantizado sobre una KV con la
forma/strides reales de vLLM (int8_per_token_head, NHD, bloques de 832), q/k/v reales
(capa 35). Compara contra atencion float y contra el camino Triton int8 de vLLM."""
import math, sys, types, time, torch
import sk18_a0bis_ptx as A
from vllm._genesis import sk18_attn as P
from vllm.v1.kv_cache_interface import KVQuantMode
dev = "cuda"; D = 256; NH = 2; G = 6; BS = 832
torch.manual_seed(0)

impl = types.SimpleNamespace(_kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD, head_size=D, num_heads=12,
                             alibi_slopes=None, sinks=None, sliding_window=(-1, -1), logits_soft_cap=0,
                             scale=1.0 / 16, chunk_lookback=-1, use_td=False, num_kv_heads=NH)
layer = types.SimpleNamespace(_k_scale=torch.ones(1, device=dev), _v_scale=torch.ones(1, device=dev), layer_name="capa35")

def kv_nuevo(nblocks):
    t = torch.zeros(nblocks, BS, NH, 520, dtype=torch.int8, device=dev)
    return t.permute(0, 2, 1, 3)       # logico (B, H, N, 520) con strides NHD

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
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
nblocks = (N + BS - 1) // BS + 6
kv = kv_nuevo(nblocks)
perm = torch.randperm(nblocks)[: (N + BS - 1) // BS].to(torch.int32)
slots = (perm[torch.arange(N) // BS].to(torch.int64) * BS + torch.arange(N) % BS).to(dev)
# escritura en tramos como un prefill (7488)
for t0 in range(0, N, 7488):
    t1 = min(N, t0 + 7488)
    P.escribir(impl, layer, k[t0:t1].half(), v[t0:t1].half(), kv, slots[t0:t1])
# referencia float y decode de las 4 ultimas posiciones (MTP)
pos = torch.arange(N - 4, N, device=dev)
ref = A.referencia(q, k, v, pos).float()                 # [4, 12, 256]
bt = perm[None, :].clone()
qsl = torch.tensor([0, 4], dtype=torch.int32)
md = meta(qsl, torch.tensor([N], dtype=torch.int32), bt, 4)
out = torch.zeros(4, 12, D, dtype=torch.float16, device=dev)
P.forward(impl, layer, q[pos].half(), kv, md, out)
err_d = ((out.float() - ref).norm(dim=-1) / ref.norm(dim=-1)).mean().item()
# prefill decuantizado (mismas 4 posiciones como si fueran prefill: max_query_len > 5 no, forzar)
out2 = torch.zeros_like(out)
import os
os.environ["GENESIS_PN131_PREFILL"] = "triton"
P._prefill_decuant(impl, layer, q[pos].half(), kv, md, out2)
out4 = torch.zeros_like(out)
os.environ["GENESIS_PN131_PREFILL"] = "flashinfer"
P._prefill_decuant(impl, layer, q[pos].half(), kv, md, out4)
print("prefill FlashInfer vs Triton (decuant) dif:", ((out4.float()-out2.float()).norm(dim=-1)/out2.float().norm(dim=-1)).mean().item())
err_p = ((out2.float() - ref).norm(dim=-1) / ref.norm(dim=-1)).mean().item()
print(f"N={N}: decode SK-18h error {100*err_d:.3f}% | prefill decuant+Triton error {100*err_p:.3f}% | dif entre caminos {100*((out.float()-out2.float()).norm(dim=-1)/out2.float().norm(dim=-1)).mean().item():.3f}%", flush=True)
# lote de 3 secuencias con largos distintos (comparten la KV) + tiempo
seqs = torch.tensor([N, N - 3000, N - 9000], dtype=torch.int32)
btl = perm[None, :].repeat(3, 1)
qsl3 = torch.tensor([0, 4, 8, 12], dtype=torch.int32)
qq = torch.cat([q[s - 4:s] for s in seqs.tolist()]).half()
md3 = meta(qsl3, seqs, btl, 12)
out3 = torch.zeros(12, 12, D, dtype=torch.float16, device=dev)
P.forward(impl, layer, qq, kv, md3, out3)
refs = torch.cat([A.referencia(q, k, v, torch.arange(s - 4, s, device=dev)).float() for s in seqs.tolist()])
err3 = ((out3.float() - refs).norm(dim=-1) / refs.norm(dim=-1)).view(3, 4, 12).mean((1, 2))
print("lote de 3: error por secuencia", [f"{100*e:.3f}%" for e in err3.tolist()], flush=True)
def medir(f, n=30):
    for _ in range(5): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e3
print(f"tiempo forward decode (1 seq, 4 tok): {medir(lambda: P.forward(impl, layer, q[pos].half(), kv, md, out)):.3f} ms | lote 3: {medir(lambda: P.forward(impl, layer, qq, kv, md3, out3)):.3f} ms | escritura 4 tok: {medir(lambda: P.escribir(impl, layer, k[N-4:N].half(), v[N-4:N].half(), kv, slots[N-4:N])):.3f} ms", flush=True)
