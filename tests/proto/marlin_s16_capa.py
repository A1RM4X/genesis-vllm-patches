"""PN130: Marlin W4A8 propio con escalas int16 CON signo, contra la referencia y contra PN125."""
import glob, sys, torch
sys.path.insert(0, "/p")
from marlin_w4a8_capa import tensor, dev, mu, ops, scalar_types
from marlin_w4a8_fix import voltear
torch.ops.load_library(glob.glob("/root/.cache/genesis/marlin_s16/genesis_marlin_s16*.so")[0])

def correr(pref):
    qw = tensor(pref + ".weight_packed").to(dev); sc = tensor(pref + ".weight_scale").to(dev)
    N, K = tensor(pref + ".weight_shape").tolist(); G = K // sc.shape[1]
    sh = torch.arange(0, 32, 4, device=dev, dtype=torch.int32)
    q = ((qw.unsqueeze(-1) >> sh) & 0xF).reshape(N, -1)[:, :K]
    W = (q - 8).float() * sc.float().repeat_interleave(G, dim=1)
    x = torch.randn(128, K, device=dev).half()
    ref = x.float() @ W.t()
    pn, pk = mu.marlin_padded_nk(N, K, G)
    e0 = mu.marlin_make_empty_g_idx(torch.device(dev)); ws = mu.marlin_make_workspace_new(torch.device(dev))
    def prep(qq, ss, signo):
        mq = ops.gptq_marlin_repack(mu.marlin_pad_qweight(qq.t().contiguous(), N, K, pn, pk), perm=e0, size_k=pk, size_n=pn, num_bits=4, is_a_8bit=True)
        s = mu.marlin_permute_scales(mu.marlin_pad_scales(ss.t().contiguous(), N, K, pn, pk, G), size_k=pk, size_n=pn, group_size=G, is_a_8bit=True)
        if signo:   # normalizar por |s|.max(): el int16 conserva el signo
            m = s.abs().max()
            gs = (m.float() / 4096)
            s = (s / m * 4096).round().to(torch.int16).view(s.dtype)
        else:
            s, gs = mu.marlin_act_int8_process_scales(s)
        return mq, s, gs
    def gemm(op, mq, s, gs):
        xa, xs = mu.marlin_quant_input(x.reshape(-1, K), torch.int8)
        xa = mu.marlin_pad_dim(xa, K, pk)
        y = op(xa, None, mq, None, s, xs * gs, None, e0, e0, e0, ws, scalar_types.uint4b8.id,
               xa.shape[0], pn, pk, True, False, True, False)
        return mu.marlin_unpad_output(y, N, pn).float()
    res = {}
    mq, s, gs = prep(qw, sc, False); res["upstream W4A8 (escalas crudas)"] = gemm(torch.ops._C.marlin_gemm, mq, s, gs)
    q2, s2 = voltear(qw, sc, N, K, G)
    mq, s, gs = prep(q2, s2, False); res["upstream W4A8 + PN125"] = gemm(torch.ops._C.marlin_gemm, mq, s, gs)
    mq, s, gs = prep(qw, sc, True);  res["PN130 s16 con signo"] = gemm(torch.ops.genesis_marlin.marlin_gemm_s16, mq, s, gs)
    print(pref.split("layers.")[1], " | ".join(f"{k}: {100*((v-ref).norm()/ref.norm()).item():.3f}%" for k, v in res.items()), flush=True)

for pref in sys.argv[1:] or ["model.language_model.layers.3.self_attn.k_proj", "model.language_model.layers.0.linear_attn.in_proj_z"]:
    correr(pref)
