"""Separa las dos piezas del W4A8 de Marlin: per_token_quant_int8 (Triton) y marlin_gemm int8."""
import sys, torch
sys.path.insert(0, "/p")
from marlin_w4a8_capa import tensor, dev, mu, ops, scalar_types
from vllm.model_executor.layers.quantization.utils.int8_utils import per_token_quant_int8

pref = sys.argv[1] if len(sys.argv) > 1 else "model.language_model.layers.3.self_attn.k_proj"
qw = tensor(pref + ".weight_packed").to(dev); sc = tensor(pref + ".weight_scale").to(dev)
N, K = tensor(pref + ".weight_shape").tolist(); G = K // sc.shape[1]
sh = torch.arange(0, 32, 4, device=dev, dtype=torch.int32)
q = ((qw.unsqueeze(-1) >> sh) & 0xF).reshape(N, -1)[:, :K]
W = (q - 8).float() * sc.float().repeat_interleave(G, dim=1)
x = torch.randn(128, K, device=dev).half()
ref = x.float() @ W.t()

# 1) cuantizacion por token
xq, xs = per_token_quant_int8(x)
xs_t = x.float().abs().amax(-1, keepdim=True) / 127
xq_t = (x.float() / xs_t).round().clamp(-127, 127).to(torch.int8)
print("per_token_quant_int8: dtype", xq.dtype, xs.dtype, xs.shape,
      "| err dequant triton", ((xq.float() * xs - x.float()).norm() / x.float().norm()).item(),
      "| iguales a torch:", (xq == xq_t).float().mean().item(), (xs - xs_t).abs().max().item(), flush=True)

# 2) GEMM con int8 hecho a mano
pn, pk = mu.marlin_padded_nk(N, K, G)
e0 = mu.marlin_make_empty_g_idx(torch.device(dev)); ws = mu.marlin_make_workspace_new(torch.device(dev))
mq = ops.gptq_marlin_repack(mu.marlin_pad_qweight(qw.t().contiguous(), N, K, pn, pk), perm=e0, size_k=pk, size_n=pn, num_bits=4, is_a_8bit=True)
s = mu.marlin_permute_scales(mu.marlin_pad_scales(sc.t().contiguous(), N, K, pn, pk, G), size_k=pk, size_n=pn, group_size=G, is_a_8bit=True)
s16, gs = mu.marlin_act_int8_process_scales(s)
print("escalas:", s.dtype, s.shape, "max", s.max().item(), "min", s.min().item(), "| gs", gs.item(), "| int16 min/max",
      s16.view(torch.int16).min().item(), s16.view(torch.int16).max().item(), flush=True)
for nombre, (a, asc) in {"triton": (xq, xs), "torch": (xq_t, xs_t)}.items():
    for gsn, g in (("gs", gs), ("gs=None", None)):
        try:
            y = ops.marlin_gemm(a, None, mq, None, s16, (asc * g) if g is not None else asc, None, e0, e0, e0, ws,
                                scalar_types.uint4b8, size_m=a.shape[0], size_n=pn, size_k=pk,
                                is_k_full=True, use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False).float()
            print(f"gemm int8 ({nombre}, {gsn}): err {100*((y-ref).norm()/ref.norm()).item():.3f}%  cos "
                  f"{torch.nn.functional.cosine_similarity(y.flatten(), ref.flatten(), dim=0).item():.4f}  ratio {(y.norm()/ref.norm()).item():.4g}", flush=True)
        except Exception as ex:
            print(nombre, gsn, "ERROR", str(ex)[:200])
