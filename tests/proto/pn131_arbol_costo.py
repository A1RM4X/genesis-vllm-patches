"""Costo de la mascara de ancestros en la atencion de PN131 (16 capas del target).

Es la pieza que faltaba del mapa del sobrecosto del arbol: GDN y conv ya estan medidos, esta no.
Compara el mismo decode uniforme con la mascara de CADENA (que es bit a bit lo de produccion) y con
la de un ARBOL, para separar cuanto cuesta la mascara en si. Con grafos CUDA.

Uso: pn131_arbol_costo.py [contexto] [pedidos]
"""
import os, sys, time, types, torch
ARB = os.environ.get("GENESIS_ENABLE_ARBOL", "1") == "1"
from vllm._genesis import sk18_attn as P
from vllm._genesis import arbol_borrador as ab
from vllm.v1.kv_cache_interface import KVQuantMode
from vllm.v1.attention.backends.triton_attn import MIN_LAUNCH_GRID_SIZE_2D, NUM_PAR_SOFTMAX_SEGMENTS

dev = "cuda"; D = 256; NH = 2; G = 6; HQ = NH * G; BS = 832
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
B = int(sys.argv[2]) if len(sys.argv) > 2 else 6
L = 9
torch.manual_seed(0)
impl = types.SimpleNamespace(_kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD, head_size=D,
                             num_heads=HQ, alibi_slopes=None, sinks=None, sliding_window=(-1, -1),
                             logits_soft_cap=0, scale=1.0 / 16, chunk_lookback=-1, use_td=False,
                             num_kv_heads=NH)
layer = types.SimpleNamespace(_k_scale=torch.ones(1, device=dev), _v_scale=torch.ones(1, device=dev),
                              layer_name="sint")
k = torch.nn.functional.normalize(torch.randn(N, NH, D, device=dev), dim=-1) * 16
v = torch.randn(N, NH, D, device=dev)
nblocks = (N + BS - 1) // BS + 6
kv = torch.zeros(nblocks, BS, NH, 520, dtype=torch.int8, device=dev).permute(0, 2, 1, 3)
perm = torch.randperm(nblocks)[: (N + BS - 1) // BS].to(torch.int32)
slots = (perm[torch.arange(N) // BS].to(torch.int64) * BS + torch.arange(N) % BS).to(dev)
for t0 in range(0, N, 7488):
    t1 = min(N, t0 + 7488)
    P.escribir(impl, layer, k[t0:t1].half(), v[t0:t1].half(), kv, slots[t0:t1])
q = torch.nn.functional.normalize(torch.randn(B * L, HQ, D, device=dev), dim=-1).half() * 16
qsl = torch.arange(0, (B + 1) * L, L, dtype=torch.int32)
md = types.SimpleNamespace(
    num_actual_tokens=B * L, max_query_len=L, query_start_loc=qsl.to(dev), max_seq_len=N,
    seq_lens=torch.full((B,), N, dtype=torch.int32, device=dev),
    block_table=perm[None, :].repeat(B, 1).contiguous().to(dev), causal=True,
    genesis_qsl_cpu=qsl.cpu(), seq_threshold_3D=MIN_LAUNCH_GRID_SIZE_2D // NH,
    num_par_softmax_segments=NUM_PAR_SOFTMAX_SEGMENTS, softmax_segm_output=None,
    softmax_segm_max=None, softmax_segm_expsum=None, mm_prefix_range_tensor=None,
    rswa_prefix_lens=None, rswa_window=None)
out = torch.zeros(B * L, HQ, D, dtype=torch.float16, device=dev)
P.forward(impl, layer, q, kv, md, out)          # crea los buffers y compila
torch.cuda.synchronize()
m = P.mascara_arbol(torch.device(dev, torch.cuda.current_device()), L) if ARB else None
cadena = ((1 << torch.arange(L, device=dev, dtype=torch.int32)) - 1).repeat(B) if ARB else None
arbol = ab.bits_ancestros(torch.tensor([[-1, 0, 1, 2, 3, 2, 1, 6, 0]]))[0].to(dev).repeat(B) if ARB else None


def cron(bits):
    if bits is not None:
        m[: B * L] = bits
    for _ in range(20):
        P.forward(impl, layer, q, kv, md, out)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        P.forward(impl, layer, q, kv, md, out)
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(200):
        g.replay()
    torch.cuda.synchronize(); return (time.time() - t0) / 200 * 1e6


print(f"contexto {N}, {B} pedidos x {L} tokens:")
if not ARB:
    # kernel de PRODUCCION: compilado SIN -DARBOL, o sea sin la mascara en el lazo interno
    print(f"  produccion (sin -DARBOL):         {cron(None):7.1f} us/capa")
else:
    c = cron(cadena); a = cron(arbol)
    print(f"  -DARBOL, mascara de cadena:       {c:7.1f} us/capa   x16 capas = {c * 16 / 1000:.2f} ms/paso")
    print(f"  -DARBOL, mascara de arbol:        {a:7.1f} us/capa   x16 = {a * 16 / 1000:.2f} ms/paso"
          f"  ({(a - c) * 16 / 1000:+.3f} ms)")
