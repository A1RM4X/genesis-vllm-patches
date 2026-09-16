// SPDX-License-Identifier: Apache-2.0
// LAB — sonda de ldmatrix (normal y transpuesto): de donde saca cada byte de cada registro.
// Shared X [8192] <- patron global (valor = f(posicion)). Cada lane L usa direccion
// X + L * 128. Salida por lane: los 4 registros (16 bytes).
#include <cuda_pipeline.h>
#ifndef OP
#define OP 0
#endif
#if OP == 0
#define LD_INS "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
#elif OP == 1
#define LD_INS "ldmatrix.sync.aligned.trans.m8n8.x4.shared.b16 "
#else
#define LD_INS "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
#endif
__device__ __forceinline__ unsigned sdir(const void* p) { return static_cast<unsigned>(__cvta_generic_to_shared(p)); }
extern "C" __global__ void __launch_bounds__(32)
lab_ldmatrix(const unsigned char* __restrict__ pat, unsigned* __restrict__ out, int paso)
{
    extern __shared__ unsigned char _smem[];
    const int tid = threadIdx.x;
    for (int v = tid; v < 8192 / 16; v += 32)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sdir(&_smem[v * 16])), "l"(pat + v * 16), "r"(16));
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    unsigned r[4];
    unsigned pa = sdir(&_smem[tid * paso]);
    asm volatile(LD_INS "{%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(pa));
    for (int i = 0; i < 4; ++i) out[tid * 4 + i] = r[i];
}
