"""SK-18i hibrido: compara (m, S, O) de las paginas ESPEJADAS (int8) contra las mismas paginas
calculadas por el camino int4, para ver si la conversion de unidades entre los dos es correcta."""
import sys, types, torch
import sk18_a0bis_ptx as A
from vllm._genesis import sk18_attn as P
from vllm.v1.kv_cache_interface import KVQuantMode
dev = "cuda"; D = 256; NH = 2; G = 6; BS = 832
torch.manual_seed(0)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
exec(open("/p/sk18i_test.py").read().split("for nombre, (qm, cont) in MODOS.items():")[0].replace(
    'N = int(sys.argv[1]) if len(sys.argv) > 1 else 5000', 'pass'))

impl = types.SimpleNamespace(_kv_quant_mode=KVQuantMode.INT4_PER_TOKEN_HEAD, head_size=D, num_heads=12,
                             alibi_slopes=None, sinks=None, sliding_window=(-1, -1), logits_soft_cap=0,
                             scale=1.0 / 16, chunk_lookback=-1, use_td=False, num_kv_heads=NH)
layer = types.SimpleNamespace(_k_scale=torch.ones(1, device=dev), _v_scale=torch.ones(1, device=dev),
                              layer_name="capa35")
kv = torch.zeros(nblocks, BS, NH, 264, dtype=torch.uint8, device=dev).permute(0, 2, 1, 3)
for t0 in range(0, N, 7488):
    t1 = min(N, t0 + 7488)
    P._md_forzada = meta(torch.tensor([0, t1 - t0], dtype=torch.int32),
                         torch.tensor([t1], dtype=torch.int32), bt, t1 - t0)
    P.escribir(impl, layer, kk[t0:t1].half(), vv[t0:t1].half(), kv, slots[t0:t1])
P._md_forzada = None
md = meta(qsl, torch.tensor([N], dtype=torch.int32), bt, 4)
out = torch.zeros(4, 12, D, dtype=torch.float16, device=dev)
R = NH * 32
NCH = (N + BS - 1) // BS

P._decode_uniforme(impl, qq[pos].half(), kv, md, out, 1, 4, False)
torch.cuda.synchronize()
bf = P._bufs[0]
hib = {k: getattr(bf, k)[: NCH * R * (D if k in ("oh", "ol") else 1)].clone() for k in ("oh", "ol", "om", "os", "oc")}
dueno = bf.dueno.clone()
# ahora el mismo decode sin espejo (todas las paginas por int4)
P.VENT = 0
P._k.clear(); P._bufs.clear()
P._decode_uniforme(impl, qq[pos].half(), kv, md, out, 1, 4, False)
torch.cuda.synchronize()
bf2 = P._bufs[0]
sol = {k: getattr(bf2, k)[: NCH * R * (D if k in ("oh", "ol") else 1)].clone() for k in ("oh", "ol", "om", "os", "oc")}

esp = [p for p in range(NCH) if int(dueno[0 * P.VENT + (p % max(P.VENT, 1))] if P.VENT else -1) == int(bt[0, p])]
print("dueno:", dueno.tolist(), "| bloques de las ultimas paginas:", bt[0, NCH - 3:NCH].tolist())
for p in [NCH - 2, NCH - 1, NCH - 3, 0]:
    r = 0
    m1 = int(hib["om"].view(NCH, R)[p, r]); m2 = int(sol["om"].view(NCH, R)[p, r])
    s1 = int(hib["os"].view(NCH, R)[p, r]); s2 = int(sol["os"].view(NCH, R)[p, r])
    o1 = (hib["oh"].view(NCH, R, D)[p, r].long() * 256 + hib["ol"].view(NCH, R, D)[p, r].long()
          - hib["oc"].view(NCH, R)[p, r].long())
    o2 = (sol["oh"].view(NCH, R, D)[p, r].long() * 256 + sol["ol"].view(NCH, R, D)[p, r].long()
          - sol["oc"].view(NCH, R)[p, r].long())
    rel = ((o1 - o2).float().norm() / o2.float().norm().clamp_min(1e-6)).item()
    print(f"pagina {p:3d} fila {r}: m hib={m1} int4={m2} (dif {m1-m2}) | S hib={s1} int4={s2} "
          f"(razon {s1/max(s2,1):.3f}) | O dif rel={100*rel:.2f}% | O[0] hib={int(o1[0])} int4={int(o2[0])}")
