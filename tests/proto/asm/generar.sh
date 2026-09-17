#!/usr/bin/env bash
# Deja el PTX y el SASS del Marlin propio (PN130) en texto plano, para leerlos a mano.
# El PTX es lo que escribe nvcc; el SASS es lo que realmente ejecuta la placa despues de ptxas,
# y NO son lo mismo: ptxas reordena, fusiona y elige instrucciones. Hay que mirar los dos.
set -euo pipefail
REPO=/home/usuario/Proyectos/genesis-vllm-patches
SALIDA=$REPO/tests/proto/asm
GEN=/root/.cache/genesis/marlin_s16/gen
IMG=vllm/vllm-openai:v0.27.1

docker run --rm -v $REPO:/w -v /home/usuario/.cache/genesis:/root/.cache/genesis \
  -v $REPO/vllm/_genesis/kernels/marlin_s16:/m -w /w --entrypoint bash $IMG -c '
set -e
FLAGS="-DTORCH_EXTENSION_NAME=genesis_marlin_s16 -DTORCH_API_INCLUDE_EXTENSION_H
  -I/root/.cache/genesis/marlin_s16/gen -I/m/inc
  -isystem /usr/local/lib/python3.12/dist-packages/torch/include
  -isystem /usr/local/lib/python3.12/dist-packages/torch/include/torch/csrc/api/include
  -isystem /usr/local/cuda/include -isystem /usr/include/python3.12
  -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
  -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__
  --expt-relaxed-constexpr --compiler-options -fPIC -O3 -std=c++17
  -DTORCH_TARGET_VERSION=0x020c000000000000 -DUSE_CUDA -DMARLIN_NAMESPACE_NAME=marlin"
cd /root/.cache/genesis/marlin_s16/gen
# Solo la unidad con las instanciaciones del kernel W4A8, no el marlin.cu entero (que arrastra torch)
/usr/local/cuda/bin/nvcc $FLAGS -arch=sm_86 -ptx sm80_kernel_s8_u4b8_float16.cu -o /w/tests/proto/asm/marlin_s16.ptx
/usr/local/cuda/bin/nvcc $FLAGS -arch=sm_86 -cubin sm80_kernel_s8_u4b8_float16.cu -o /w/tests/proto/asm/marlin_s16.cubin
'
# nvdisasm no viene en la imagen de vLLM; el cuobjdump del host si
/usr/local/cuda/bin/cuobjdump -sass $SALIDA/marlin_s16.cubin > $SALIDA/marlin_s16.sass
rm -f $SALIDA/marlin_s16.cubin
wc -l $SALIDA/marlin_s16.ptx $SALIDA/marlin_s16.sass
