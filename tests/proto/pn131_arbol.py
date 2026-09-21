"""PN131 + ARBOL offline: (a) con la mascara de CADENA la salida es bit a bit la del kernel de
siempre; (b) con un arbol coincide con atencion float bajo la misma mascara de ancestros.
Uso: pn131_arbol.py <modo: base|cadena|arbol> <salida.pt> [L] [N]   (un proceso por modo)"""
import math, os, sys, types, torch
modo, salida = sys.argv[1], sys.argv[2]
L = int(sys.argv[3]) if len(sys.argv) > 3 else 9
N = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
os.environ["GENESIS_ENABLE_PN131_SK18"] = "1"; os.environ["GENESIS_PN131_MAXTOK"] = str(L)
os.environ["GENESIS_ENABLE_ARBOL"] = "0" if modo == "base" else "1"
from vllm._genesis import sk18_attn as P
from vllm.v1.kv_cache_interface import KVQuantMode
dev = "cuda"; D = 256; NH = 2; G = 6; HQ = NH * G; BS = 832
torch.manual_seed(0)
impl = types.SimpleNamespace(_kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD, head_size=D, num_heads=HQ,
                             alibi_slopes=None, sinks=None, sliding_window=(-1, -1), logits_soft_cap=0,
                             scale=1.0 / 16, chunk_lookback=-1, use_td=False, num_kv_heads=NH)
layer = types.SimpleNamespace(_k_scale=torch.ones(1, device=dev), _v_scale=torch.ones(1, device=dev), layer_name="sint")
k = torch.nn.functional.normalize(torch.randn(N, NH, D, device=dev), dim=-1) * 16
v = torch.randn(N, NH, D, device=dev)
q = torch.nn.functional.normalize(torch.randn(N, HQ, D, device=dev) + 0.6 * k.repeat_interleave(G, 1), dim=-1) * 16
# Prueba DECISIVA de la mascara: las queries nuevas apuntan fuerte a las keys nuevas (logit ~16
# contra ~0 del contexto), asi que la salida es casi el promedio de v sobre las keys nuevas
# VISIBLES: una mascara equivocada da un error de orden 1, no de decimas de punto.
if os.environ.get("FUERTE", "0") == "1":
    u = torch.nn.functional.normalize(torch.randn(NH, D, device=dev), dim=-1) * 16
    k[N - L:] = u[None]
    q[N - L:] = u.repeat_interleave(G, 0)[None]
nblocks = (N + BS - 1) // BS + 6
kv = torch.zeros(nblocks, BS, NH, 520, dtype=torch.int8, device=dev).permute(0, 2, 1, 3)
perm = torch.randperm(nblocks)[: (N + BS - 1) // BS].to(torch.int32)
slots = (perm[torch.arange(N) // BS].to(torch.int64) * BS + torch.arange(N) % BS).to(dev)
for t0 in range(0, N, 7488):
    t1 = min(N, t0 + 7488); P.escribir(impl, layer, k[t0:t1].half(), v[t0:t1].half(), kv, slots[t0:t1])
pos = torch.arange(N - L, N, device=dev)
# arbol fijo sobre los L-1 tokens nuevos (indice 0 = ancla). padre[j] en indices de token (0 = ancla)
g = torch.Generator().manual_seed(5)
padre = [-1, 0] + [int(torch.randint(0, j, (1,), generator=g)) for j in range(2, L)]
if modo != "arbol":
    padre = [-1] + list(range(0, L - 1))                       # cadena
vis = torch.zeros(L, L, dtype=torch.bool)
for j in range(L):
    a = j
    while a >= 0:
        vis[j, a] = True; a = padre[a]
# bits: el token i (i>=1) ocupa el bit i-1; el ancla (0) no lleva bit, es "contexto"
anc = torch.tensor([sum(1 << (i - 1) for i in range(1, L) if vis[j, i]) for j in range(L)], dtype=torch.int32, device=dev)
from vllm.v1.attention.backends.triton_attn import MIN_LAUNCH_GRID_SIZE_2D, NUM_PAR_SOFTMAX_SEGMENTS
qsl = torch.tensor([0, L], dtype=torch.int32)
md = types.SimpleNamespace(num_actual_tokens=L, max_query_len=L, query_start_loc=qsl.to(dev), max_seq_len=N,
    seq_lens=torch.tensor([N], dtype=torch.int32, device=dev), block_table=perm[None, :].clone().to(dev), causal=True,
    genesis_qsl_cpu=qsl.cpu(), seq_threshold_3D=MIN_LAUNCH_GRID_SIZE_2D // NH, num_par_softmax_segments=NUM_PAR_SOFTMAX_SEGMENTS,
    softmax_segm_output=None, softmax_segm_max=None, softmax_segm_expsum=None,
    mm_prefix_range_tensor=None, rswa_prefix_lens=None, rswa_window=None)
out = torch.zeros(L, HQ, D, dtype=torch.float16, device=dev)
if modo == "arbol":
    # La mascara vive en UN buffer estatico (los grafos CUDA hornean la direccion), no en el
    # metadata: se escribe ahi, como hace arbol_runner antes del forward del target.
    P._get_bufs(torch.device(dev, torch.cuda.current_device()), NH, kv.shape[2], G)
    P.mascara_arbol(torch.device(dev, torch.cuda.current_device()), L)[:L] = anc
P.forward(impl, layer, q[pos].half(), kv, md, out); torch.cuda.synchronize()
# referencia float con la MISMA visibilidad: contexto completo + ancestros entre los nuevos
lg = torch.einsum("bkgd,tkd->bkgt", q[pos].view(L, NH, G, D), k) / math.sqrt(D)
m = torch.ones(L, N, dtype=torch.bool, device=dev); m[:, N - L:] = vis.to(dev)
ref = torch.einsum("bkgt,tkd->bkgd", lg.masked_fill(~m[:, None, None, :], float("-inf")).softmax(-1), v).reshape(L, HQ, D)
err = ((out.float() - ref).norm(dim=-1) / ref.norm(dim=-1))
# sensibilidad: la MISMA salida contra la referencia con la mascara equivocada (causal simple)
mc = torch.ones(L, N, dtype=torch.bool, device=dev); mc[:, N - L:] = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))
refc = torch.einsum("bkgt,tkd->bkgd", lg.masked_fill(~mc[:, None, None, :], float("-inf")).softmax(-1), v).reshape(L, HQ, D)
errc = ((out.float() - refc).norm(dim=-1) / refc.norm(dim=-1))
print(f"        contra la mascara CAUSAL (equivocada si hay arbol): medio {100*errc.mean():.3f}%  por token {[round(100*float(x),2) for x in errc.mean(1)]}")
print(f"{modo:7s} L={L} padres={padre} | error vs float: medio {100*err.mean():.3f}%  max {100*err.max():.3f}%  por token {[round(100*float(x),2) for x in err.mean(1)]}")
torch.save(out.cpu(), salida)
