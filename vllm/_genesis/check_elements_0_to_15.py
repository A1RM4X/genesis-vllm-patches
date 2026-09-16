import os
import torch
from safetensors import safe_open
from vllm._genesis.kernels.requant_inplace import (
    _quant_inplace_triton, _PTQ_KERNEL, _amax_por_bloque_kernel
)

snap_dir = "/root/.cache/huggingface/hub/models--orcarouter--Qwen3.8-27B-Uncensored-FP8/snapshots/0787858da83e6640e289c0c22d092d92f4e97fdb"
if not os.path.exists(snap_dir):
    base = "/root/.cache/huggingface/hub/models--orcarouter--Qwen3.8-27B-Uncensored-FP8/snapshots"
    snap_dir = os.path.join(base, os.listdir(base)[0])
file_path = os.path.join(snap_dir, "model-00001-of-00007.safetensors")

with safe_open(file_path, framework="pt", device="cuda:0") as f:
    w = f.get_tensor("model.language_model.layers.0.mlp.down_proj.weight")
    s = f.get_tensor("model.language_model.layers.0.mlp.down_proj.weight_scale_inv")
    w0 = w[:, 0:8704].contiguous()
    s0 = s[:, 0:68].contiguous()

    n, k = w0.shape
    bk, bn = 128, 128
    kb_n, nb_n = k // bk, n // bn

    bits_a = w0.clone().view(torch.int8)
    bits_b = w0.clone().view(torch.int8)

    print("Initial bytes [0, :16]:", [b & 0xFF for b in bits_a[0, :16].tolist()])

    amax_nk = torch.empty((n, kb_n), dtype=torch.float32, device="cuda:0")
    _amax_por_bloque_kernel[(n, kb_n)](
        bits_a, s0, amax_nk,
        bits_a.stride(0), s0.stride(0), s0.stride(1), amax_nk.stride(0),
        BN_BLK=bn, BK=bk, num_warps=4, num_stages=1,
    )
    s_row = (amax_nk.amax(dim=1) / 127.0).clamp(min=1e-30)
    need = torch.log2((amax_nk / (127.0 * s_row[:, None])).clamp(min=1e-30))
    shift = torch.ceil(need.reshape(nb_n, bn, kb_n).amax(dim=1)).clamp(-10.0, 10.0)

    _quant_inplace_triton[(n, kb_n)](
        bits_a, bits_a, s0, s_row, shift,
        bits_a.stride(0), s0.stride(0), s0.stride(1),
        shift.stride(0), shift.stride(1),
        BN_BLK=bn, BK=bk, num_warps=4, num_stages=1,
    )

    _PTQ_KERNEL((n, kb_n), bits_b, bits_b, s0, s_row, shift,
                bits_b.stride(0), s0.stride(0), s0.stride(1),
                shift.stride(0), shift.stride(1), bn, bk)

    torch.cuda.synchronize()
    print("Triton result [0, :16]:", bits_a[0, :16].tolist())
    print("PTX    result [0, :16]:", bits_b[0, :16].tolist())

