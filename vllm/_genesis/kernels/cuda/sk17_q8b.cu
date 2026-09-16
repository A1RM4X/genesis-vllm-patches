// SPDX-License-Identifier: Apache-2.0
//
// SK-17-Q8b — int8 por token en dos lanzamientos con UN HILO POR BLOQUE de 256
// (la version de un hilo por fila recorria 5120 elementos en serie: 20x mas lenta
// que scaled_int8_quant).
//
//   sk17_q8_max:   bmx[m*NG + g] = max |x| del bloque (fp32)
//   sk17_q8_quant: fila m: mx = max_g bmx, sx[m] = mx/127, cuantiza SOLO su bloque g

#include <cuda_fp16.h>
#include <cuda_pipeline.h>

#ifndef NWARPS
#define NWARPS 8
#endif
#define GB 256

extern "C" __global__ void __launch_bounds__(NWARPS * 32)
sk17_q8_max(const __half* __restrict__ x, float* __restrict__ bmx, int M, int K)
{
    const int NG = K / GB;
    const long long celda = (long long)blockIdx.x * (NWARPS * 32) + threadIdx.x;
    const long long m = celda / NG;
    if (m >= M) return;
    const int g = (int)(celda % NG);
    const size_t base = (size_t)m * K + (size_t)g * GB;
    __half2 mx2 = __float2half2_rn(0.0f);
    for (int k = 0; k < GB; k += 2) {
        const __half2 v = *reinterpret_cast<const __half2*>(x + base + k);
        mx2 = __hmax2(mx2, __hmax2(v, __hneg2(v)));
    }
    const float a = (float)__low2half(mx2), b = (float)__high2half(mx2);
    bmx[(size_t)m * NG + g] = a > b ? a : b;
}

extern "C" __global__ void __launch_bounds__(NWARPS * 32)
sk17_q8_quant(const __half* __restrict__ x, const float* __restrict__ bmx,
              signed char* __restrict__ xq, float* __restrict__ sx, int M, int K)
{
    const int NG = K / GB;
    const long long celda = (long long)blockIdx.x * (NWARPS * 32) + threadIdx.x;
    const long long m = celda / NG;
    if (m >= M) return;
    const int g = (int)(celda % NG);
    float mx = 1e-8f;
    const size_t fb = (size_t)m * NG;
    for (int j = 0; j < NG; ++j) mx = bmx[fb + j] > mx ? bmx[fb + j] : mx;
    if (g == 0) sx[m] = mx / 127.0f;
    const float inv = 127.0f / mx;
    const size_t base = (size_t)m * K + (size_t)g * GB;
    for (int k = 0; k < GB; k += 2) {
        const __half2 v = *reinterpret_cast<const __half2*>(x + base + k);
        int a = __float2int_rn((float)__low2half(v) * inv);
        int b = __float2int_rn((float)__high2half(v) * inv);
        a = a > 127 ? 127 : (a < -127 ? -127 : a);
        b = b > 127 ? 127 : (b < -127 ? -127 : b);
        xq[base + k]     = (signed char)a;
        xq[base + k + 1] = (signed char)b;
    }
}
