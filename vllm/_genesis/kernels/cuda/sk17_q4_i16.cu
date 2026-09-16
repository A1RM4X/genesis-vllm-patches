// SPDX-License-Identifier: Apache-2.0
//
// SK-17-Q4 — int16 (salida de la Hadamard INT8 de SK-17) -> int4 por grupo + nibbles.
//
//   mx      = max_i |y[m, g*G + i]|                    (entero)
//   q[m,i]  = clamp(round(y[m,i] * 7 / mx), -7, 7)
//   s[m,g]  = mx * sx[m] / (16 * 7)  (escala int8 de x y 1/sqrt(256))
//
// Un hilo = un grupo. El maximo es entero; el unico flotante por elemento es la
// multiplicacion por 7/mx antes del redondeo.

#include <cuda_pipeline.h>

#ifndef NWARPS
#define NWARPS 1
#endif

extern "C" __global__ void __launch_bounds__(NWARPS * 32)
sk17_q4_i16(const short* __restrict__ y, const float* __restrict__ sx,
            signed char* __restrict__ q, float* __restrict__ s, int M, int K, int G)
{
    const int NG = K / G;
    const long long celda = (long long)blockIdx.x * (NWARPS * 32) + threadIdx.x;
    const long long m = celda / NG;
    if (m >= M) return;
    const int g = (int)(celda % NG);
    const size_t base = (size_t)m * K + (size_t)g * G;

    int mx = 1;
    for (int i = 0; i < G; ++i) {
        int v = y[base + i];
        v = v < 0 ? -v : v;
        mx = v > mx ? v : mx;
    }
    const float inv = 7.0f / (float)mx;
    // sx / 16: escala int8 de x y el 1/sqrt(256) de la Hadamard.
    s[(size_t)m * NG + g] = (float)mx * sx[m] / 112.0f;

    const size_t qbase = base >> 1;
    for (int i = 0; i < G; i += 2) {
        int qa = __float2int_rn((float)y[base + i] * inv);
        int qb = __float2int_rn((float)y[base + i + 1] * inv);
        qa = qa > 7 ? 7 : (qa < -7 ? -7 : qa);
        qb = qb > 7 ? 7 : (qb < -7 ? -7 : qb);
        q[qbase + (i >> 1)] = (signed char)((qa & 15) | ((qb & 15) << 4));
    }
}
