"""Valida el arreglo de escalas negativas para Marlin W4A8-INT8: s -> |s|, q -> 16 - q (clamp 15)."""
import sys, torch
sys.path.insert(0, "/p")
from marlin_w4a8_capa import tensor, dev, mu, ops, scalar_types


def voltear(qw, sc, N, K, G):
    """qw [N, K/8] int32 empaquetado (uint4b8), sc [N, K/G] -> version con escalas >= 0."""
    sh = torch.arange(0, 32, 4, device=qw.device, dtype=torch.int32)
    q = ((qw.unsqueeze(-1) >> sh) & 0xF).reshape(N, -1)            # [N, 8*K/8] (incluye relleno)
    neg = (sc < 0).repeat_interleave(G, dim=1)
    pad = q.shape[1] - neg.shape[1]
    if pad:
        neg = torch.nn.functional.pad(neg, (0, pad))
    q = torch.where(neg, (16 - q).clamp(max=15), q)
    qw2 = (q.reshape(N, -1, 8) << sh).sum(-1, dtype=torch.int64)
    qw2 = torch.where(qw2 >= 2**31, qw2 - 2**32, qw2).to(torch.int32)
    return qw2, sc.abs()


def correr(pref):
    qw = tensor(pref + ".weight_packed").to(dev); sc = tensor(pref + ".weight_scale").to(dev)
    N, K = tensor(pref + ".weight_shape").tolist(); G = K // sc.shape[1]
    sh = torch.arange(0, 32, 4, device=dev, dtype=torch.int32)
    q = ((qw.unsqueeze(-1) >> sh) & 0xF).reshape(N, -1)[:, :K]
    W = (q - 8).float() * sc.float().repeat_interleave(G, dim=1)
    qw2, sc2 = voltear(qw, sc, N, K, G)
    q2 = ((qw2.unsqueeze(-1) >> sh) & 0xF).reshape(N, -1)[:, :K]
    W2 = (q2 - 8).float() * sc2.float().repeat_interleave(G, dim=1)
    x = torch.randn(128, K, device=dev).half()
    ref = x.float() @ W.t()
    pn, pk = mu.marlin_padded_nk(N, K, G)
    e0 = mu.marlin_make_empty_g_idx(torch.device(dev)); ws = mu.marlin_make_workspace_new(torch.device(dev))
    out = {}
    for nombre, (qq, ss, a8) in {"W4A16 original": (qw, sc, False), "W4A8 original": (qw, sc, True),
                                 "W4A8 volteado": (qw2, sc2, True), "W4A16 volteado": (qw2, sc2, False),
                                 "W4A8 signo_absmax": (qw, sc, "abs")}.items():
        modo = a8; a8 = bool(a8)
        mq = ops.gptq_marlin_repack(mu.marlin_pad_qweight(qq.t().contiguous(), N, K, pn, pk), perm=e0, size_k=pk, size_n=pn, num_bits=4, is_a_8bit=a8)
        s = mu.marlin_permute_scales(mu.marlin_pad_scales(ss.t().contiguous(), N, K, pn, pk, G), size_k=pk, size_n=pn, group_size=G, is_a_8bit=a8)
        gs = None
        if modo == "abs":
            m = s.abs().max()
            gs = 1 / 4096 * m.float()
            s = (s / m * 4096).round().to(torch.int16).view(s.dtype)
        elif a8:
            s, gs = mu.marlin_act_int8_process_scales(s)
        y = mu.apply_gptq_marlin_linear(input=x, weight=mq, weight_scale=s, weight_zp=e0, g_idx=e0, g_idx_sort_indices=e0,
                workspace=ws, wtype=scalar_types.uint4b8, output_size_per_partition=N, input_size_per_partition=K,
                is_k_full=True, input_global_scale=gs, bias=None, input_dtype=torch.int8 if a8 else None).float()
        out[nombre] = 100 * ((y - ref).norm() / ref.norm()).item()
    dW = 100 * ((W2 - W).norm() / W.norm()).item()
    print(f"{pref.split('layers.')[1]:28s} peso volteado vs original {dW:.3f}% | " +
          " | ".join(f"{k} {v:.3f}%" for k, v in out.items()), flush=True)


if __name__ == "__main__":
  for pref in sys.argv[1:] or ["model.language_model.layers.3.self_attn.k_proj",
                               "model.language_model.layers.0.linear_attn.in_proj_z",
                               "model.language_model.layers.0.linear_attn.out_proj"]:
      correr(pref)
