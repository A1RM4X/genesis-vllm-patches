import math, torch
import sk18_a0bis_ptx as A
dev = "cuda"; D = 256; G = 6; BS = 832
UMBRAL = 16 * math.log(2)          # 16 medios-log = peso 0 exacto en el kernel
for capa in (3, 7, 11, 35):
    try:
        q, k, v = A.cargar(capa)
    except Exception as e:
        print(capa, "sin volcado"); continue
    Nt = k.shape[0]
    for N in (8000, 16000, Nt):
        pos = torch.arange(N - 4, N, device=dev)
        fr = []
        for h in range(2):
            Q = q[pos][:, h * G:(h + 1) * G].reshape(-1, D).float(); K = k[:N, h].float()
            lg = (Q @ K.t()) / 16.0
            npag = (N + BS - 1) // BS
            pm = torch.nn.functional.pad(lg, (0, npag * BS - N), value=-1e9).view(-1, npag, BS).amax(-1)
            mg = pm.amax(1, keepdim=True)
            fr.append((pm < mg - UMBRAL).float().mean().item())
            # unidades de 64 keys
            nu = (N + 63) // 64
            um = torch.nn.functional.pad(lg, (0, nu * 64 - N), value=-1e9).view(-1, nu, 64).amax(-1)
            fr.append((um < mg - UMBRAL).float().mean().item())
        print(f"capa {capa} N={N}: paginas nulas {100*(fr[0]+fr[2])/2:.1f}%  unidades nulas {100*(fr[1]+fr[3])/2:.1f}%", flush=True)
