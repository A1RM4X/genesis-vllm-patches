"""Busca que parte del camino W4A8-INT8 de Marlin (v0.27.1) rompe: combina variantes
de repack / permutacion / proceso de escalas y mide contra x @ W^T."""
import itertools, sys, torch
sys.path.insert(0, "/p")
from marlin_w4a8_capa import tensor, dev, mu, ops, scalar_types

pref = sys.argv[1] if len(sys.argv) > 1 else "model.language_model.layers.3.self_attn.k_proj"
qw = tensor(pref + ".weight_packed").to(dev); sc = tensor(pref + ".weight_scale").to(dev)
N, K = tensor(pref + ".weight_shape").tolist(); G = K // sc.shape[1]
sh = torch.arange(0, 32, 4, device=dev, dtype=torch.int32)
q = ((qw.unsqueeze(-1) >> sh) & 0xF).reshape(N, -1)[:, :K]
W = (q - 8).float() * sc.float().repeat_interleave(G, dim=1)
x = torch.randn(128, K, device=dev).half()
ref = x.float() @ W.t()
pn, pk = mu.marlin_padded_nk(N, K, G)
e0 = mu.marlin_make_empty_g_idx(torch.device(dev)); ws = mu.marlin_make_workspace_new(torch.device(dev))
xor8 = torch.tensor(0x88888888 - 2**32, dtype=torch.int32, device=dev)
for flip, rep8, perm8, proc in itertools.product((False, True), (True,), (True, False), ("vllm", "x8", "x1/8")):
    qsrc = qw ^ xor8 if flip else qw
    qg = mu.marlin_pad_qweight(qsrc.t().contiguous(), N, K, pn, pk)
    mq = ops.gptq_marlin_repack(qg, perm=e0, size_k=pk, size_n=pn, num_bits=4, is_a_8bit=rep8)
    s = mu.marlin_permute_scales(mu.marlin_pad_scales(sc.t().contiguous(), N, K, pn, pk, G), size_k=pk, size_n=pn, group_size=G, is_a_8bit=perm8)
    gs = torch.tensor(1.0, device=dev)
    if proc == "vllm":
        s, gs = mu.marlin_act_int8_process_scales(s)
    elif proc == "x8":
        s, gs = mu.marlin_act_int8_process_scales(s); gs = gs * 8
    elif proc == "x1/8":
        s, gs = mu.marlin_act_int8_process_scales(s); gs = gs / 8
    try:
        y = mu.apply_gptq_marlin_linear(input=x, weight=mq, weight_scale=s, weight_zp=e0, g_idx=e0, g_idx_sort_indices=e0,
                workspace=ws, wtype=scalar_types.uint4b8, output_size_per_partition=N, input_size_per_partition=K,
                is_k_full=True, input_global_scale=gs, bias=None, input_dtype=torch.int8).float()
        err = ((y - ref).norm() / ref.norm()).item()
        cos = torch.nn.functional.cosine_similarity(y.flatten(), ref.flatten(), dim=0).item()
        ratio = (y.norm() / ref.norm()).item()
        print(f"flip={flip!s:5} repack_a8={rep8!s:5} permute_a8={perm8!s:5} escalas={proc:17s} err={100*err:10.3f}%  cos={cos:.4f}  |y|/|ref|={ratio:.4g}", flush=True)
    except Exception as ex:
        print(f"repack_a8={rep8} permute_a8={perm8} escalas={proc}: ERROR {str(ex)[:120]}")
