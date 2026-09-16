import collections, torch
from vllm._genesis.kernels.ptx_lab import Kernel
dev = "cuda"
import sys
pos = torch.arange(8192, device=dev)
for op, nom in [((0, "normal"), (1, "trans"), (2, "x4.trans"))[int(sys.argv[1])]]:
    try:
        k = Kernel("lab_ldmatrix.cu", "lab_ldmatrix", defs=[f"-DOP={op}"], warps=1)
        res = []
        for pat in ((pos & 255), (pos >> 8)):
            out = torch.zeros(32 * 4, dtype=torch.int32, device=dev)
            k.lanzar((1, 1), [pat.to(torch.uint8).contiguous(), out, 128], shared=8192, sync=True)
            b = out.view(32, 4).to(torch.int64) & 0xffffffff
            bytes_ = torch.stack([(b >> (8 * i)) & 255 for i in range(4)], -1)   # [32 lanes, 4 regs, 4 bytes]
            res.append(bytes_)
        src = res[0] + (res[1] << 8)                                             # offset fuente
        print(f"== {nom}")
        for L in (0, 1, 2, 3, 4, 8, 16, 17):
            filas = []
            for r in range(4):
                offs = src[L, r].tolist()
                filas.append("r%d:%s" % (r, ",".join(f"L{o // 128}+{o % 128}" for o in offs)))
            print(f"  lane {L:2d}: " + "  ".join(filas))
    except Exception as e:
        print(nom, "FALLA", str(e)[-300:])
