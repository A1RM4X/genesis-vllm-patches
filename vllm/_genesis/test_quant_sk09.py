"""quant_per_token (SK-09) contra referencia torch, en los K y M reales."""
import torch
from vllm._genesis.kernels.sk09_norm_embed import quant_per_token

torch.manual_seed(0)
print(f"{'K':>7}{'M':>6}{'cos':>11}{'esc':>9}{'max_rel':>10}  veredicto")
malos = 0
for K in (3072, 5120, 7168, 8704, 17408):
    for M in (1, 4, 5, 16, 17, 40, 130, 512, 1200):
        x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda") * 3.0
        q, s = quant_per_token(x)
        s = s.reshape(-1)
        # referencia: amax por fila / 127, redondeo al mas cercano
        amax = x.to(torch.float32).abs().amax(dim=1).clamp_min(1e-30)
        s_ref = amax / 127.0
        q_ref = (x.to(torch.float32) / s_ref[:, None]).round().clamp(-127, 127)
        rec = q.to(torch.float32) * s[:, None]
        ref = q_ref * s_ref[:, None]
        cos = torch.nn.functional.cosine_similarity(rec.flatten(), ref.flatten(), dim=0).item()
        esc = (rec.abs().mean() / ref.abs().mean()).item()
        rel = ((rec - ref).abs().max() / ref.abs().max()).item()
        ok = cos > 0.9999 and 0.99 < esc < 1.01 and rel < 0.05
        if not ok:
            malos += 1
        print(f"{K:>7}{M:>6}{cos:>11.6f}{esc:>9.4f}{rel:>10.4f}  {'OK' if ok else '*** ROTO ***'}")
print("\nROTOS:", malos)
