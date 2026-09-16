// SPDX-License-Identifier: Apache-2.0
//
// SK-15-ACT — cuantizacion int4 + empaquetado en nibbles de las activaciones
// (ya rotadas) para el GEMM s4 de SK-14 / SK-15. En una sola pasada por grupo:
//
//     s[m,g]  = max_i |x[m, g*G + i]| / 7
//     q[m,i]  = clamp(round(x[m,i] / s), -7, 7)
//     byte j  = q[2j] & 0xF  |  (q[2j+1] & 0xF) << 4
//
// Cada hilo procesa UN grupo (celda lineal = bloque*32 + hilo -> (m, g)). Con
// G = K (una escala por token) cada hilo es una fila entera.
//
// El grupo se lee dos veces (max, despues cuantizar): la segunda lectura cae en
// la L1 porque la primera la acaba de traer. Pares de elementos en __half2:
// una carga de 32 bits trae los dos nibbles de un byte de salida.

#include <cuda_fp16.h>
#include <cuda_pipeline.h>

#ifndef NWARPS
#define NWARPS 1
#endif

extern "C" __global__ void __launch_bounds__(NWARPS * 32)
sk15_act_q4(const __half* __restrict__ x, signed char* __restrict__ q,
            float* __restrict__ s, int M, int K, int G)
{
    const int NG = K / G;
    const long long celda = (long long)blockIdx.x * (NWARPS * 32) + threadIdx.x;
    const long long m = celda / NG;
    if (m >= M) return;
    const int g = (int)(celda % NG);
    const size_t base = (size_t)m * K + (size_t)g * G;

    // Pasada 1: maximo del valor absoluto, de a dos elementos.
    __half2 mx2 = __float2half2_rn(0.0f);
    for (int i = 0; i < G; i += 2) {
        const __half2 v = *reinterpret_cast<const __half2*>(x + base + i);
        mx2 = __hmax2(mx2, __hmax2(v, __hneg2(v)));
    }
    float mx = (float)__low2half(mx2);
    const float mh = (float)__high2half(mx2);
    if (mh > mx) mx = mh;
    if (mx < 1e-8f) mx = 1e-8f;
    const float esc = mx / 7.0f;
    const float inv = 7.0f / mx;
    s[(size_t)m * NG + g] = esc;

    // Pasada 2: cuantizar y empaquetar. round-to-nearest con clamp a [-7, 7].
    const signed char* dummy = 0; (void)dummy;
    const size_t qbase = base >> 1;
    for (int i = 0; i < G; i += 2) {
        const __half2 v = *reinterpret_cast<const __half2*>(x + base + i);
        const float a = (float)__low2half(v) * inv;
        const float b = (float)__high2half(v) * inv;
        int qa = __float2int_rn(a);
        int qb = __float2int_rn(b);
        qa = qa > 7 ? 7 : (qa < -7 ? -7 : qa);
        qb = qb > 7 ? 7 : (qb < -7 ? -7 : qb);
        q[qbase + (i >> 1)] = (signed char)((qa & 15) | ((qb & 15) << 4));
    }
}
