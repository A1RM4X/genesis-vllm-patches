"""Compila el Marlin W4A8 con escalas int16 con signo (PN130) como extension propia."""
import glob, os, subprocess, sys, time
import torch
from torch.utils import cpp_extension
AQUI = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(AQUI, "src")
BUILD = os.environ.get("GENESIS_MARLIN_S16_BUILD", "/root/.cache/genesis/marlin_s16")
os.makedirs(BUILD, exist_ok=True)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
t0 = time.time()
gen = os.path.join(BUILD, "gen"); os.makedirs(gen, exist_ok=True)
# el generador escribe al lado de si mismo: copiar las fuentes a gen/
for f in glob.glob(os.path.join(SRC, "*")):
    subprocess.check_call(["cp", f, gen])
subprocess.check_call([sys.executable, os.path.join(gen, "generate_kernels.py"), "8.6"])
# Una sola unidad de traduccion: las instanciaciones explicitas de Marlin<...>
# salen con enlace local en el .o del kernel y el link de la .so no las ve.
kernels = sorted(glob.glob(os.path.join(gen, "sm80_kernel_*.cu")))
with open(os.path.join(gen, "marlin.cu"), "a") as f:
    for k in kernels:
        f.write(f'\n#include "{os.path.basename(k)}"\n')
fuentes = [os.path.join(gen, "marlin.cu")]
print("fuentes:", [os.path.basename(f) for f in fuentes], flush=True)
mod = cpp_extension.load(
    name="genesis_marlin_s16", sources=fuentes, build_directory=BUILD,
    extra_include_paths=[gen, os.path.join(AQUI, "inc")],
    extra_cflags=["-O3", "-std=c++17", "-DTORCH_TARGET_VERSION=0x020c000000000000", "-DUSE_CUDA"],
    extra_cuda_cflags=["-O3", "-std=c++17", "-DTORCH_TARGET_VERSION=0x020c000000000000", "-DUSE_CUDA",
                       "-DMARLIN_NAMESPACE_NAME=marlin", "-Xcompiler", "-fvisibility=default"]
    + (["-DGENESIS_QSERVE_CRUDO"] if os.environ.get("GENESIS_QSERVE_CRUDO") == "1" else []),
    is_python_module=False, verbose=True)
print(f"listo en {time.time()-t0:.0f}s:", hasattr(torch.ops.genesis_marlin, "marlin_gemm_s16"))
