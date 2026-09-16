import sys, math, torch
sys.argv = ["x"]
import sk18_a0bis_ptx as A
dev = "cuda"
for capa in (3, 35):
    q, k, v = A.cargar(capa)
    N = k.shape[0]
    pos = torch.linspace(8000, N - 1, 48, device=dev).long()
    B = len(pos); G = 6; h = 0
    T = int(pos.max()) + 1
    qq = q[pos][:, :G]
    lg_ref = torch.einsum("bgd,td->bgt", qq, k[:T, h]) / 16
    mask = torch.arange(T, device=dev)[None, :] > pos[:, None]
    def salida(lg, V):
        w = lg.masked_fill(mask[:, None, :], float("-inf")).softmax(-1)
        return torch.einsum("bgt,td->bgd", w, V[:T])
    ref = salida(lg_ref, v[:, h])
    e = lambda o: 100 * ((o - ref).norm(dim=-1) / ref.norm(dim=-1)).mean().item()
    qr = A.rotar_ptx(q[pos]); kr = A.rotar_ptx(k)
    q8, sq = A.q8_ptx(qr); k8, sk = A.q8_ptx(kr); v8, sv = A.q8_ptx(v)
    Qh = q8[:, :G].reshape(-1, 256); sqh = sq[:, :G].reshape(-1)
    z = A.qk_ptx(Qh, k8[:, h]).view(B, G, N)[:, :, :T]
    lg_q = z.double() * sqh.view(B, G, 1).double() * sk[None, None, :T, h].double() / 16
    print(f"capa {capa}")
    print("  1 Q8K8 (PTX) logits float, softmax float, V float :", round(e(salida(lg_q.float(), v[:, h])), 3))
    Vq = v8[:, h].float() * sv[:, h, None]
    print("  2 + V int8                                          :", round(e(salida(lg_q.float(), Vq)), 3))
    skmax = sk[:, h].max(); skf = torch.round(sk[:, h] / skmax * 32767).to(torch.int64)
    zp = ((z.to(torch.int64) >> 8) * skf[None, None, :T]) >> 11
    alpha = 256.0 * skmax * sqh.view(B, G, 1) / 16 / 16.0
    lg_zp = zp.double() * alpha.double()
    print("  3 + logits enteros z' (>>8, skf, >>11)              :", round(e(salida(lg_zp.float(), Vq)), 3))
    lg_m = lg_zp.masked_fill(mask[:, None, :], -1e9)
    zm = lg_m.amax(-1, keepdim=True)
    dreal = (zm - lg_zp).clamp(min=0)
    idx = torch.round(dreal * 1023 / 16).clamp(max=1023)
    wl = torch.exp(-idx * 16 / 1023).masked_fill(mask[:, None, :], 0)
    o4 = torch.einsum("bgt,td->bgd", wl.float() / wl.sum(-1, keepdim=True).float(), Vq[:T])
    print("  4 + indice de tabla (1024, c=16) exp exacta          :", round(e(o4), 3))
    wq = torch.round(32639 * wl).masked_fill(mask[:, None, :], 0)
    o5 = torch.einsum("bgt,td->bgd", wq.float() / wq.sum(-1, keepdim=True).float(), Vq[:T])
    print("  5 + pesos 15 bits relativos al max                   :", round(e(o5), 3))
    svf = torch.round(sv[:, h] / sv[:, h].max() * 32767)
    w2 = torch.floor(wq * svf[None, None, :T] / 32768)
    o6 = torch.einsum("bgt,td->bgd", w2.float(), v8[:T, h].float()) * sv[:, h].max() / wq.sum(-1, keepdim=True).float()
    print("  6 + svf plegada en el peso (floor)                   :", round(e(o6), 3))
    w2r = torch.floor((wq * svf[None, None, :T] + 16384) / 32768)
    o7 = torch.einsum("bgt,td->bgd", w2r.float(), v8[:T, h].float()) * sv[:, h].max() / wq.sum(-1, keepdim=True).float()
    print("  7 = 5 + svf plegada con redondeo                     :", round(e(o7), 3))
    for bs in (128, 832):
        vb = v[:T, h]
        nb = (T + bs - 1) // bs
        vp = torch.nn.functional.pad(vb, (0, 0, 0, nb * bs - T)).view(nb, bs, 256)
        sb = vp.abs().amax((1, 2), keepdim=True).clamp_min(1e-8) / 127
        vqb = ((vp / sb).round().clamp(-127, 127) * sb).view(-1, 256)[:T]
        o8 = torch.einsum("bgt,td->bgd", wq.float() / wq.sum(-1, keepdim=True).float(), vqb)
        print(f"  8 = 5 con V int8 escala por bloque {bs:4d} (sin svf)     :", round(e(o8), 3))
