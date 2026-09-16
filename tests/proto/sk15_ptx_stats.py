import re, collections
from vllm._genesis.kernels.ptx_lab import Kernel
def stats(nombre, k):
    t = k.ptx(forzar=True)
    ins = re.findall(r'^\s+([a-z][a-z0-9]*(?:\.[a-z0-9_:]+)*)\s', t, re.M)
    base = collections.Counter(i.split('.')[0] for i in ins)
    mma = sum(1 for i in ins if i.startswith('mma'))
    ld = collections.Counter(i for i in ins if i.startswith('ld'))
    print(f"== {nombre}: total {len(ins)} | mma {mma} | " + ", ".join(f"{a}:{b}" for a, b in base.most_common(10)))
    print("   ld:", dict(ld.most_common(6)), "| sync/wait:", len(re.findall(r'barrier|syncthreads|arrive|wait', t)))
    return t
stats("SK-14 32w", Kernel("sk14_gateup_w4a4.cu", "sk14_gateup_w4a4", defs=["-DS4=1","-DNWARPS=32","-DWM=32","-DWN=16"], warps=32))
t3 = stats("SK-15 diag3", Kernel("sk15_gateup_w4a4g.cu", "sk15_gateup_w4a4g", defs=["-DSK15_DIAG=3"], warps=32))
stats("SK-15 diag1", Kernel("sk15_gateup_w4a4g.cu", "sk15_gateup_w4a4g", defs=["-DSK15_DIAG=1"], warps=32))
open("/tmp/sk15_diag3.ptx","w").write(t3)
