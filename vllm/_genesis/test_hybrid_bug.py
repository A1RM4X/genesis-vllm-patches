"""Reproduce el fallo: int8_hybrid_gemm interpreta w_shifts fp32 (escalas por
bloque) con la semantica vieja de shift diadico entero."""
import torch
from vllm._genesis.kernels.int8_hybrid_gemm import int8_hybrid_gemm

torch.manual_seed(0)
M, K, N = 8, 512, 256
Bk = Bn = 128

w = torch.randn((K, N), dtype=torch.float32, device="cuda") * 0.02
wt = w.unflatten(0, (K // Bk, Bk)).unflatten(2, (N // Bn, Bn)).permute(0, 2, 1, 3)
amax = wt.abs().amax(dim=(2, 3))
bs = torch.where(amax > 1e-12, amax / 127.0, torch.ones_like(amax))
w_i8 = (wt / bs.unsqueeze(-1).unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
w_i8 = w_i8.permute(0, 2, 1, 3).reshape(K, N).contiguous()
w_deq = (w_i8.unflatten(0, (K // Bk, Bk)).unflatten(2, (N // Bn, Bn)).permute(0, 2, 1, 3).float()
         * bs.unsqueeze(-1).unsqueeze(-1)).permute(0, 2, 1, 3).reshape(K, N)

x = torch.randn((M, K), dtype=torch.float32, device="cuda")
a_sc = x.abs().amax(dim=1) / 127.0
a_i8 = (x / a_sc[:, None]).round().clamp(-127, 127).to(torch.int8)

ref = (a_i8.float() * a_sc[:, None]) @ w_deq
b_col = w_i8.t().contiguous().t()
ones = torch.ones(N, dtype=torch.float32, device="cuda")
got = int8_hybrid_gemm(a_i8, b_col, a_sc, ones, bs, torch.float32)

cos = torch.nn.functional.cosine_similarity(ref.flatten(), got.flatten(), dim=0).item()
print(f"cos={cos:.6f}  ref_absmax={ref.abs().max():.4f}  got_absmax={got.abs().max():.4f}")
print("VEREDICTO:", "OK" if cos > 0.999 else "ROTO")

# --- Diseno C: shift diadico entero, tiene que seguir andando ---
sh = torch.randint(-3, 1, (K // Bk, N // Bn), dtype=torch.int8, device="cuda")
b_ch = torch.rand(N, dtype=torch.float32, device="cuda") * 0.01 + 1e-3
ref_c = ((a_i8.float() * a_sc[:, None]) @
         (w_i8.float().unflatten(0, (K // Bk, Bk)).unflatten(2, (N // Bn, Bn)).permute(0, 2, 1, 3)
          * torch.exp2(sh.float()).unsqueeze(-1).unsqueeze(-1)).permute(0, 2, 1, 3).reshape(K, N)
         ) * b_ch[None, :]
got_c = int8_hybrid_gemm(a_i8, b_col, a_sc, b_ch, sh, torch.float32)
cos_c = torch.nn.functional.cosine_similarity(ref_c.flatten(), got_c.flatten(), dim=0).item()
rel_c = ((ref_c - got_c).abs().max() / ref_c.abs().max()).item()
print(f"diadico: cos={cos_c:.6f} rel={rel_c:.2e} -> {'OK' if cos_c > 0.9999 and rel_c < 1e-5 else 'ROTO'}")
