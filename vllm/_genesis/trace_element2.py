import torch

# Let's decode byte from bits_a[0, 2]
b = 0xAA # example
# Let's inspect real byte from checkpoint
import os
from safetensors import safe_open
from vllm._genesis.kernels.requant_inplace import _amax_por_bloque_kernel

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

    bits = w0.clone().view(torch.int8)
    byte_val = bits[0, 2].item() & 0xFF
    s_val = s0[0, 0].item()

    n, k = w0.shape
    bk, bn = 128, 128
    kb_n, nb_n = k // bk, n // bn
    amax_nk = torch.empty((n, kb_n), dtype=torch.float32, device="cuda:0")
    _amax_por_bloque_kernel[(n, kb_n)](
        bits, s0, amax_nk,
        bits.stride(0), s0.stride(0), s0.stride(1), amax_nk.stride(0),
        BN_BLK=bn, BK=bk, num_warps=4, num_stages=1,
    )
    s_row = (amax_nk.amax(dim=1) / 127.0).clamp(min=1e-30)
    need = torch.log2((amax_nk / (127.0 * s_row[:, None])).clamp(min=1e-30))
    shift = torch.ceil(need.reshape(nb_n, bn, kb_n).amax(dim=1)).clamp(-10.0, 10.0)

    s_row_0 = s_row[0].item()
    shift_0 = shift[0, 0].item()

    # Manual decode
    sign = -1.0 if (byte_val & 0x80) else 1.0
    exp = (byte_val >> 3) & 0x0F
    mant = byte_val & 0x07
    m_int = mant + (8 if exp != 0 else 0)
    v_val = sign * m_int * (2.0 ** (max(exp, 1) - 10))

    paso = s_row_0 * (2.0 ** shift_0)
    x = v_val * s_val / paso
    q_expected = int(round(x))

    print(f"Byte: {byte_val:#04x} ({byte_val})")
    print(f"s_val (scale_inv[0,0]): {s_val}")
    print(f"s_row[0]: {s_row_0}")
    print(f"shift[0,0]: {shift_0}")
    print(f"v_val: {v_val}")
    print(f"paso: {paso}")
    print(f"x: {x}")
    print(f"q_expected: {q_expected}")

