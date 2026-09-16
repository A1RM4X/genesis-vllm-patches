"""Valida vllm/_genesis/marlin_w4a8_escalas.positivizar sobre una capa real con el camino
completo (permute_param_layout_ + MarlinLinearKernel) contra la referencia dequantizada."""
import sys, torch
sys.path.insert(0, "/p")
from marlin_w4a8_capa import tensor, dev, mu, ops, scalar_types
from vllm.model_executor.parameter import PackedvLLMParameter, GroupQuantScaleParameter
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import MPLinearLayerConfig
from vllm.model_executor.kernels.linear.mixed_precision.marlin import MarlinLinearKernel
from vllm._genesis import marlin_w4a8_escalas as P
import vllm.model_executor.parameter as _par
_par.get_tensor_model_parallel_rank = lambda: 0
_par.get_tensor_model_parallel_world_size = lambda: 1

pref = sys.argv[1] if len(sys.argv) > 1 else "model.language_model.layers.3.self_attn.k_proj"
qw = tensor(pref + ".weight_packed").to(dev); sc = tensor(pref + ".weight_scale").to(dev)
N, K = tensor(pref + ".weight_shape").tolist(); G = K // sc.shape[1]
sh = torch.arange(0, 32, 4, device=dev, dtype=torch.int32)
q = ((qw.unsqueeze(-1) >> sh) & 0xF).reshape(N, -1)[:, :K]
W = (q - 8).float() * sc.float().repeat_interleave(G, dim=1)
x = torch.randn(128, K, device=dev).half()
ref = x.float() @ W.t()
for modo in ("sin PN125", "con PN125"):
    for a8 in (False, True):
        layer = torch.nn.Module()
        noop = lambda *a, **k: None
        layer.weight_packed = PackedvLLMParameter(data=qw.clone(), input_dim=1, output_dim=0, packed_dim=1, packed_factor=8, weight_loader=noop)
        layer.weight_scale = GroupQuantScaleParameter(data=sc.clone(), input_dim=1, output_dim=0, weight_loader=noop)
        cfg = MPLinearLayerConfig(full_weight_shape=(K, N), partition_weight_shape=(K, N), weight_type=scalar_types.uint4b8,
                                  act_type=torch.int8 if a8 else torch.float16, group_size=G, zero_points=False, has_g_idx=False)
        kern = MarlinLinearKernel(cfg, w_q_param_name="weight_packed", w_s_param_name="weight_scale")
        if modo == "con PN125":
            P.positivizar(layer, "weight_packed", "weight_scale", cfg)
        kern.process_weights_after_loading(layer)
        y = kern.apply_weights(layer, x).float()
        print(f"{modo:10s} {'W4A8' if a8 else 'W4A16'}: err {100*((y-ref).norm()/ref.norm()).item():.3f}%", flush=True)
print("stats PN125: capas", P.STATS.capas, "neg%", 100 * P.STATS.negativos / max(1, P.STATS.grupos), "err peso%", 100 * (P.STATS.err2 / max(P.STATS.ref2, 1e-30)) ** 0.5)
