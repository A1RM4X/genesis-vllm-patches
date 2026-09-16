"""Valida los kernels de vllm._genesis.gdn_cinta contra upstream.

Upstream: K+1 columnas de estado por request. Cinta: una columna + cinta.
1) forward spec en 40 pasos con aceptaciones al azar: salida igual.
2) materializacion: estado[src] + bias filas == columna src+bias de upstream.
"""
import types, torch
from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update as ref)
import vllm._genesis.gdn_cinta as g

torch.manual_seed(1)
dev, dt = "cuda", torch.float16
H, HV, K, V, SPEC, N = 8, 24, 128, 128, 3, 5
T = SPEC + 1
page_extra = 1000                      # simula stride de pagina > estado
S = 1 + N * T
raw = torch.randn(S, HV * V * K + page_extra, device=dev, dtype=dt) * 0.05
h_ref = raw[:, :HV * V * K].view(S, HV, V, K)
raw2 = raw.clone(); h_cin = raw2[:, :HV * V * K].view(S, HV, V, K)
assert h_cin.stride(0) != HV * V * K
A_log = torch.randn(HV, device=dev) * 0.5; dt_bias = torch.randn(HV, device=dev) * 0.5
cols = torch.arange(1, S, device=dev, dtype=torch.int32).view(N, T)
g._init_slots(8, dev)
layer = types.SimpleNamespace(tp_size=2, num_k_heads=16, num_v_heads=48, head_k_dim=K,
                              head_v_dim=V, num_spec=SPEC, prefix="t")
g.enlazar(layer, dev)
slots = torch.tensor([3, 7, 1, 12, 5], device=dev, dtype=torch.int32)
cu = torch.arange(0, (N + 1) * T, T, device=dev, dtype=torch.int32)
acc = torch.ones(N, device=dev, dtype=torch.int32)
peor = 0
for paso in range(40):
    q = torch.randn(1, N * T, H, K, device=dev, dtype=dt); k = torch.randn_like(q)
    v = torch.randn(1, N * T, HV, V, device=dev, dtype=dt)
    a = torch.randn(N * T, HV, device=dev, dtype=dt); b = torch.randn_like(a)
    o_r, _ = ref(A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=q, k=k, v=v, initial_state=h_ref,
                 inplace_final_state=True, cu_seqlens=cu, ssm_state_indices=cols,
                 num_accepted_tokens=acc, use_qk_l2norm_in_kernel=True)
    o_c, _ = g.spec_update(layer, A_log, a, b, dt_bias, q, k, v, h_cin, cu, cols, acc, slots)
    peor = max(peor, ((o_r.float() - o_c.float()).norm() / o_r.float().norm()).item())
    acc = torch.randint(1, T + 1, (N,), device=dev, dtype=torch.int32)
print(f"forward spec: error relativo maximo {peor:.2e}")

# materializacion: tras el ultimo paso, para bias 1..3 comparar con upstream col+bias
bt = torch.zeros(N, 8, dtype=torch.int32, device=dev)
bt[:, :T] = cols
dst_blocks = torch.arange(S, S + N, device=dev)  # no hay; usar bloques libres: reservo extra
raw3 = torch.zeros(S + N, raw2.shape[1], device=dev, dtype=dt); raw3[:S] = raw2
h3 = raw3[:, :HV * V * K].view(S + N, HV, V, K)
bt[:, 5] = torch.arange(S, S + N, device=dev, dtype=torch.int32)
ctx = types.SimpleNamespace(mamba_group_ids=[0], block_size=832,
                            block_table_ptrs=torch.tensor([bt.data_ptr()], dtype=torch.int64, device=dev),
                            block_table_stride_req=bt.stride(0))
g._meta.ssm_addrs = torch.tensor([h3.data_ptr()], dtype=torch.int64, device=dev)
g._meta.ssm_strides = torch.tensor([h3.stride(0) * 2], dtype=torch.int64, device=dev)
g._meta.grupos = torch.tensor([0], dtype=torch.int32, device=dev)
g._meta.cinta_addrs = torch.tensor([layer._g122_cinta.data_ptr()], dtype=torch.int64, device=dev)
g._meta.capas = [layer]; g._meta.dims = (H, HV, K, V, SPEC, g.fila(layer))
g._slots_gpu[:N] = slots
bias = torch.randint(1, T, (N,), device=dev, dtype=torch.int32)
src = torch.zeros(N, device=dev, dtype=torch.int32)
dst = torch.full((N,), 5, device=dev, dtype=torch.int32)
g._lanzar(False, ctx, N, dst, dst, dst, dst, dst, src, bias)
peor = 0
for i in range(N):
    ref_state = h_ref[cols[i, bias[i]]].float(); got = h3[S + i].float()
    peor = max(peor, ((ref_state - got).norm() / ref_state.norm()).item())
print(f"materializacion: error relativo maximo {peor:.2e} (bias {bias.tolist()})")
