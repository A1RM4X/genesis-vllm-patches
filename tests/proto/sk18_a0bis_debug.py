import sys, torch, math
sys.argv = ["x"]
import sk18_a0bis_ptx as A
dev = "cuda"
q, k, v = A.cargar(3)
N = k.shape[0]
pos = torch.linspace(8000, N - 1, 4, device=dev).long()
qr = A.rotar_ptx(q[pos]); kr = A.rotar_ptx(k)
print("rot q ok:", ((qr @ torch.eye(256, device=dev)).norm() / q[pos].norm()).item(), "err rot:", ((qr - q[pos] @ A.R.float().to(dev).t()).norm() / q[pos].norm()).item())
q8, sq = A.q8_ptx(qr); k8, sk = A.q8_ptx(kr); v8, sv = A.q8_ptx(v)
print("q8 err:", ((q8.float() * sq[..., None] - qr).norm() / qr.norm()).item(), "k8 err:", ((k8.float() * sk[..., None] - kr).norm() / kr.norm()).item(), "sq", sq.mean().item(), "sk", sk.mean().item())
h = 0; G = 6
Qh = q8[:, :G].reshape(-1, 256); M = Qh.shape[0]
z = A.qk_ptx(Qh, k8[:, h])
zref = (Qh.double() @ k8[:, h].double().t())
print("qk exacto:", bool((z.double() == zref).all()), "z max", z.max().item())
posh = pos.repeat_interleave(G)
skh = sk[:, h]; skmax = skh.max()
skf = torch.round(skh / skmax * 32767).to(torch.int64)
zp = ((z.to(torch.int64) >> 8) * skf[None, :]) >> 15
causal = torch.arange(N, device=dev)[None, :] > posh[:, None]
zp = zp.masked_fill(causal, A.PAD)
alpha = 256.0 * skmax * sq[:, :G].reshape(-1) / 16
print("alpha", alpha.mean().item(), "zp max", zp.max().item(), "logit max aprox", (zp.max(1).values * alpha).mean().item())
mq = torch.round(alpha * 1023 / A.C_RECORTE * 65536).to(torch.int32)
print("mq", mq.float().mean().item())
zm = zp.amax(1)
d = (zm[:, None] - zp).clamp(min=0)
idx = ((d * mq[:, None].to(torch.int64)) >> 16).clamp(max=1023)
keep = ~causal
lsum = A.LUT.to(torch.int64)[idx].masked_fill(~keep, 0).sum(1)
print("lsum", lsum.float().mean().item(), "idx<1023 frac", (idx < 1023).float().mean().item())
invS = ((32000 << 16) // lsum.clamp_min(1)).to(torch.int32)
print("invS", invS.float().mean().item())
# referencia float de la cabeza 0
T = int(pos.max()) + 1
qq = q[pos][:, :G]
lg = torch.einsum("bgd,td->bgt", qq, k[:T, h]) / 16
mask = torch.arange(T, device=dev)[None, :] > pos[:, None]
w = lg.masked_fill(mask[:, None, :], float("-inf")).softmax(-1)
ref = torch.einsum("bgt,td->bgd", w, v[:T, h]).reshape(-1, 256)
# emulacion float de pesos enteros
wv = (A.LUT.to(torch.int64)[idx] * invS[:, None].to(torch.int64)) >> 16
wv = wv.masked_fill(~keep, 0)
o_em = (wv.double() @ (v8[:, h].double() * sv[:, h, None].double())) / wv.sum(1, keepdim=True).double()
print("err emulacion pesos enteros vs float:", ((o_em.float() - ref).norm() / ref.norm()).item())

# kernel SK-18b denso, misma entrada
CH = 1024; K = ((N + CH - 1) // CH) * CH
Vt = torch.zeros(256, K, dtype=torch.int8, device=dev); Vt[:, :N] = v8[:, h].t()
svmax = sv[:, h].max(); svf = torch.zeros(K, dtype=torch.int16, device=dev); svf[:N] = torch.round(sv[:, h] / svmax * 32767).to(torch.int16)
Szp = torch.full((M, K), A.PAD, dtype=torch.int32, device=dev); Szp[:, :N] = zp.to(torch.int32)
out = torch.zeros(K // CH, M, 256, dtype=torch.int32, device=dev)
grid = ((256 + 63) // 64, ((M + 63) // 64) * (K // CH))
A.k18b.lanzar(grid, [Szp.contiguous(), zm.to(torch.int32).contiguous(), mq.contiguous(), invS.contiguous(), A.LUT, svf.contiguous(), Vt.contiguous(), out, M, 256, K, CH, K // CH], shared=A.SH_B)
oi = out.to(torch.int64).sum(0)
w2 = (wv * svf[:N][None, :].to(torch.int64)) >> 15
hi = (w2 + 128) >> 8; lo = w2 - (hi << 8)
ref_int = (hi.double() @ v8[:, h].double()) * 256 + lo.double() @ v8[:, h].double()
print("kernel vs ref entera exacto:", bool((oi.double() == ref_int).all()), "norma kernel", oi.double().norm().item(), "norma ref", ref_int.norm().item())
o_k = oi.double() * (svmax.double() / 32767.0) / wv.sum(1, keepdim=True).double()
print("err kernel desescalado vs float:", ((o_k.float() - ref).norm() / ref.norm()).item())
