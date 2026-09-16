"""SK-18i: cuanto satura el formato int4 (KM de 12 bits, svf, zp de 4 bits) y cuanto error
aporta cada saturacion. Lee las escalas que escribio el kernel en el pool."""
import sys, types, torch
import sk18_a0bis_ptx as A
from vllm._genesis import sk18_attn as P
from vllm.v1.kv_cache_interface import KVQuantMode
dev = "cuda"; D = 256; NH = 2; BS = 832
torch.manual_seed(0)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 57000
CAPA = int(sys.argv[2]) if len(sys.argv) > 2 else 35

q, k, v = A.cargar(CAPA); Nt = k.shape[0]
reps = (N + Nt - 1) // Nt
kk = k.repeat(reps, 1, 1)[:N].contiguous(); vv = v.repeat(reps, 1, 1)[:N].contiguous()
impl = types.SimpleNamespace(_kv_quant_mode=KVQuantMode.INT4_PER_TOKEN_HEAD, head_size=D, num_heads=12,
                             alibi_slopes=None, sinks=None, sliding_window=(-1, -1), logits_soft_cap=0,
                             scale=1.0 / 16, chunk_lookback=-1, use_td=False, num_kv_heads=NH)
layer = types.SimpleNamespace(_k_scale=torch.ones(1, device=dev), _v_scale=torch.ones(1, device=dev),
                              layer_name=f"capa{CAPA}")
nb = (N + BS - 1) // BS + 2
kv = torch.zeros(nb, BS, NH, 264, dtype=torch.uint8, device=dev).permute(0, 2, 1, 3)
slots = torch.arange(N, device=dev, dtype=torch.int64)
for t0 in range(0, N, 7488):
    P.escribir(impl, layer, kk[t0:min(N, t0 + 7488)].half(), vv[t0:min(N, t0 + 7488)].half(), kv,
               slots[t0:min(N, t0 + 7488)])
torch.cuda.synchronize()
nbq, nh, bs, blk, raw = P._geom(kv)
KOFF = bs * nh * 128
esc = raw[:, 2 * KOFF: 2 * KOFF + bs * nh * 8].reshape(nbq, bs, nh, 8)
pag = (N + bs - 1) // bs
e = esc[:pag].reshape(-1, 8).int()
w = e[:, 0] | (e[:, 1] << 8)
KM = w & 4095
vzp = ((e[:, 1] >> 4) & 15) - 8
svf = e[:, 6] | (e[:, 7] << 8)
r = torch.stack([(e[:, 2 + g] & 15) + 1 for g in range(4)], 1)
zp = torch.stack([((e[:, 2 + g] >> 4) & 15) - 8 for g in range(4)], 1)
val = torch.arange(e.shape[0], device=dev) % bs < bs          # todas las filas escritas
tot = e.shape[0]
print(f"capa {CAPA} N={N} refs={P._capa(impl, torch.device('cuda')).refs.tolist()} token-cabezas={tot}")
print(f"  KM: min={int(KM.min())} p50={int(KM.median())} max={int(KM.max())} "
      f"saturados(4095)={(KM == 4095).float().mean().item()*100:.2f}%  en 1..15={(KM < 16).float().mean().item()*100:.2f}%")
print(f"  svf: p50={int(svf.median())} max={int(svf.max())} saturados(2^VSH)={(svf >= (1 << P.VSH)).float().mean().item()*100:.2f}%")
print(f"  r por grupo: p50={int(r.median())} en 16={(r == 16).float().mean().item()*100:.1f}% en 1={(r == 1).float().mean().item()*100:.2f}%")
print(f"  zp K: min={int(zp.min())} max={int(zp.max())} en -8={(zp == -8).float().mean().item()*100:.2f}% "
      f"en 7={(zp == 7).float().mean().item()*100:.2f}%")
print(f"  zp V: min={int(vzp.min())} max={int(vzp.max())} en -8={(vzp == -8).float().mean().item()*100:.2f}% "
      f"en 7={(vzp == 7).float().mean().item()*100:.2f}%")
