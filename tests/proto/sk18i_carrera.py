"""SK-18i: aisla en QUE buffer aparece el no-determinismo del camino int4."""
import sys, types, torch
import sk18_a0bis_ptx as A
from vllm._genesis import sk18_attn as P
from vllm.v1.kv_cache_interface import KVQuantMode
dev = "cuda"; D = 256; NH = 2; G = 6; BS = 832
torch.manual_seed(0)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 57000
exec(open("/p/sk18i_test.py").read().split("for nombre, (qm, cont) in MODOS.items():")[0].replace(
    'N = int(sys.argv[1]) if len(sys.argv) > 1 else 5000', 'pass'))

impl = types.SimpleNamespace(_kv_quant_mode=KVQuantMode.INT4_PER_TOKEN_HEAD, head_size=D, num_heads=12,
                             alibi_slopes=None, sinks=None, sliding_window=(-1, -1), logits_soft_cap=0,
                             scale=1.0 / 16, chunk_lookback=-1, use_td=False, num_kv_heads=NH)
layer = types.SimpleNamespace(_k_scale=torch.ones(1, device=dev), _v_scale=torch.ones(1, device=dev),
                              layer_name="capa35")
kv = torch.zeros(nblocks, BS, NH, 264, dtype=torch.uint8, device=dev).permute(0, 2, 1, 3)
for t0 in range(0, N, 7488):
    P.escribir(impl, layer, kk[t0:min(N, t0 + 7488)].half(), vv[t0:min(N, t0 + 7488)].half(), kv,
               slots[t0:min(N, t0 + 7488)])
kv0 = kv.clone()
md = meta(qsl, torch.tensor([N], dtype=torch.int32), bt, 4)
out = torch.zeros(4, 12, D, dtype=torch.float16, device=dev)

# 1) la escritura: misma KV dos veces?
kv2 = torch.zeros_like(kv.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
for t0 in range(0, N, 7488):
    P.escribir(impl, layer, kk[t0:min(N, t0 + 7488)].half(), vv[t0:min(N, t0 + 7488)].half(), kv2,
               slots[t0:min(N, t0 + 7488)])
torch.cuda.synchronize()
print("escritura identica:", bool((kv0 == kv2).all()))

P._decode_uniforme(impl, qq[pos].half(), kv, md, out, 1, 4, False)
torch.cuda.synchronize()
bf = P._bufs[0]
R = NH * 32
NCH = (N + BS - 1) // BS
snap = lambda: {k: getattr(bf, k)[: NCH * R * (D if k in ("oh", "ol") else 1)].clone()
                for k in ("oh", "om", "os", "oc")}
s0 = snap(); q0 = bf.Q[: R * 2 * 128].clone(); sq0 = bf.sq[: R * 2 * 4].clone(); o0 = out.clone()
for i in range(8):
    out.zero_()
    P._decode_uniforme(impl, qq[pos].half(), kv, md, out, 1, 4, False)
    torch.cuda.synchronize()
    s1 = snap()
    dif = [k for k in s0 if not bool((s0[k] == s1[k]).all())]
    print(f"corrida {i}: buffers distintos={dif} Q igual={bool((q0 == bf.Q[: R*2*128]).all())} "
          f"sq igual={bool((sq0 == bf.sq[: R*2*4]).all())} salida igual={bool((o0 == out).all())}")
    if "oh" in dif:
        a = s0["oh"].view(NCH, R, D); b = s1["oh"].view(NCH, R, D)
        d = (a != b)
        idx = d.nonzero()[:4].tolist()
        print("   difs:", d.sum().item(), "primeras:", idx,
              "valores:", [(int(a[p, r, c]), int(b[p, r, c])) for p, r, c in idx],
              "m/S/C:", [(int(s0["om"].view(NCH, R)[p, r]), int(s0["os"].view(NCH, R)[p, r]),
                          int(s0["oc"].view(NCH, R)[p, r])) for p, r, _ in idx])
