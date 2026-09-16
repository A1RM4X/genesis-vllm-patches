// SPDX-License-Identifier: Apache-2.0
//
// SK-17-Q8 — cuantizacion int8 simetrica por token de x fp16 (entrada de SK-17).
//
//   sx[m]    = max_k |x[m, k]| / 127
//   xq[m, k] = clamp(round(x[m, k] / sx[m]), -127, 127)
//
// Un hilo = una fila. Pares en __half2: una carga de 32 bits trae dos elementos.

#include <cuda_fp16.h>
#include <cuda_pipeline.h>

#ifndef NWARPS
#define NWARPS 8
#endif

extern "C" __global__ void __launch_bounds__(NWARPS * 32)
sk17_q8(const __half* __restrict__ x, signed char* __restrict__ xq,
        float* __restrict__ sx, int M, int K)
{
    const long long m = (long long)blockIdx.x * (NWARPS * 32) + threadIdx.x;
    if (m >= M) return;
    const size_t base = (size_t)m * K;

    __half2 mx2 = __float2half2_rn(0.0f);
    for (int k = 0; k < K; k += 2) {
        const __half2 v = *reinterpret_cast<const __half2*>(x + base + k);
        mx2 = __hmax2(mx2, __hmax2(v, __hneg2(v)));
    }
    float mx = (float)__low2half(mx2);
    const float mh = (float)__high2half(mx2);
    if (mh > mx) mx = mh;
    if (mx < 1e-8f) mx = 1e-8f;
    sx[m] = mx / 127.0f;
    const float inv = 127.0f / mx;

    for (int k = 0; k < K; k += 2) {
        const __half2 v = *reinterpret_cast<const __half2*>(x + base + k);
        int a = __float2int_rn((float)__low2half(v) * inv);
        int b = __float2int_rn((float)__high2half(v) * inv);
        a = a > 127 ? 127 : (a < -127 ? -127 : a);
        b = b > 127 ? 127 : (b < -127 ? -127 : b);
        xq[base + k]     = (signed char)a;
        xq[base + k + 1] = (signed char)b;
    }
}
