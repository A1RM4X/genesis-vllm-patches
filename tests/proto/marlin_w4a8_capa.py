"""Microtest: Marlin W4A8-INT8 contra la referencia dequantizada, con una capa real de noon-at-cgn.

Reproduce el camino de `MarlinLinearKernel` (repack, permutacion de escalas,
`marlin_act_int8_process_scales`, `apply_gptq_marlin_linear` con input_dtype=int8)
y lo compara con W4A16 (misma capa, act fp16) y con x @ W^T en float32.
"""
import glob, json, struct, sys
import torch
from vllm.model_executor.layers.quantization.utils import marlin_utils as mu
from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types

torch.manual_seed(0)
dev = "cuda"
D = glob.glob("/root/.cache/huggingface/hub/models--noon-at-cgn--Qwen3.8-27B-Uncensored-W4A16-AutoRound/snapshots/*/")[0]
idx = json.load(open(D + "model.safetensors.index.json"))["weight_map"]


def tensor(nombre):
    f = D + idx[nombre]
    with open(f, "rb") as h:
        n = struct.unpack("<Q", h.read(8))[0]
        H = json.loads(h.read(n))
        v = H[nombre]
        s, e = v["data_offsets"]
        h.seek(8 + n + s)
        buf = bytearray(h.read(e - s))
    dt = {"I32": torch.int32, "F16": torch.float16, "BF16": torch.bfloat16, "I64": torch.int64}[v["dtype"]]
    return torch.frombuffer(buf, dtype=dt).reshape(v["shape"]).clone()


def construir(pref, a8):
    qw = tensor(pref + ".weight_packed").to(dev)          # [N, K/8] int32
    sc = tensor(pref + ".weight_scale").to(dev)           # [N, K/G] fp16
    N, K = tensor(pref + ".weight_shape").tolist()
    G = K // sc.shape[1]
    # referencia float32: uint4b8 -> q - 8
    # por tandas de filas: la GPU solo tiene ~600 MiB libres con el server arriba
    sh = torch.arange(0, 32, 4, device=dev, dtype=torch.int32)
    W = torch.empty(N, K, dtype=torch.float16, device=dev)
    for lo in range(0, N, 512):
        q = ((qw[lo:lo + 512].unsqueeze(-1) >> sh) & 0xF).reshape(-1, qw.shape[1] * 8)[:, :K]
        W[lo:lo + 512] = ((q - 8).half() * sc[lo:lo + 512].repeat_interleave(G, dim=1))
    pn, pk = mu.marlin_padded_nk(N, K, G)
    # marlin: qweight layout [K/8 cols packed] -> gptq espera [packed rows]; compressed-tensors
    # guarda [N, K/8] (packed_dim=1), MarlinLinearKernel lo permuta a input_dim=0 -> [K/8, N].t()
    qw_g = qw.t().contiguous()                               # [K/8, N]
    qw_g = mu.marlin_pad_qweight(qw_g, N, K, pn, pk)
    perm = torch.empty(0, dtype=torch.int, device=dev)
    mq = ops.gptq_marlin_repack(qw_g, perm=perm, size_k=pk, size_n=pn, num_bits=4, is_a_8bit=a8)
    s = mu.marlin_pad_scales(sc.t().contiguous(), N, K, pn, pk, G)
    s = mu.marlin_permute_scales(s, size_k=pk, size_n=pn, group_size=G, is_a_8bit=a8)
    gscale = None
    if a8 and K // G > 1:
        s, gscale = mu.marlin_act_int8_process_scales(s)
    ws = mu.marlin_make_workspace_new(torch.device(dev))
    e = mu.marlin_make_empty_g_idx(torch.device(dev))

    def f(x):
        return mu.apply_gptq_marlin_linear(
            input=x, weight=mq, weight_scale=s, weight_zp=e, g_idx=e, g_idx_sort_indices=e,
            workspace=ws, wtype=scalar_types.uint4b8, output_size_per_partition=N,
            input_size_per_partition=K, is_k_full=True, input_global_scale=gscale,
            bias=None, input_dtype=torch.int8 if a8 else None)
    return W, f, gscale


def main():
    for pref in sys.argv[1:] or ["model.language_model.layers.3.mlp.gate_proj",
                                 "model.language_model.layers.3.self_attn.q_proj",
                                 "model.language_model.layers.0.linear_attn.in_proj_qkv"]:
        W, f16, _ = construir(pref, False)
        _, f8, gs = construir(pref, True)
        K = W.shape[1]
        for nombre, x in (("normal", torch.randn(256, K, device=dev)),
                          ("outliers x50", torch.randn(256, K, device=dev) * torch.where(torch.rand(K, device=dev) < 0.01, 50.0, 1.0))):
            x = x.half()
            ref = torch.cat([x[i:i + 64].float() @ W.float().t() for i in range(0, 1)] if False else [(x[i:i + 64] @ W.t()).float() for i in range(0, x.shape[0], 64)])
            y16 = f16(x).float(); y8 = f8(x).float()
            e = lambda y: ((y - ref).norm() / ref.norm()).item()
            print(f"{pref.split('layers.')[1]:28s} {nombre:12s} W4A16 err {100*e(y16):7.3f}%  W4A8 err {100*e(y8):9.3f}%  "
                  f"nan8={torch.isnan(y8).any().item()} inf8={torch.isinf(y8).any().item()} gscale={None if gs is None else float(gs):.3g}",
                  flush=True)


if __name__ == "__main__":
    main()
