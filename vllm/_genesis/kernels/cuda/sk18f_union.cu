// SPDX-License-Identifier: Apache-2.0
//
// SK-18f/union — une los tramos del streaming entero sin torch. Todo entero:
//   mg = max_c m[c, r]
//   sc_c = 2^-(mg - m_c) en Q15 (misma cuadratica que el kernel)
//   O[r, d] = sum_c (O_c * sc_c + 2^14) >> 15   con O_c = 256 * hi + lo  (int64)
//   S[r]    = sum_c (S_c * sc_c + 2^14) >> 15
// v2: grilla 2D (query r, grupo de CPG tramos) para que escale con la cantidad de
// tramos (v1 era un bloque por query: 0,068 ms con 112 tramos, mas que la mitad del
// kernel de streaming). Cada bloque escribe su parcial en O[g, r, d] / S[g, r]; la
// suma de los grupos va afuera (una suma int64).
// Hilos: 32 lanes = 32 bloques de 8 dims; el lane 0 hace S.
#include <cuda_pipeline.h>

#ifndef QA
#define QA 22474
#endif
#ifndef QB
#define QB 6090
#endif

__device__ __forceinline__ int pot2neg_q15(int t) {
    const int n = t >> 8, f = t & 255;
    const int g = 32768 - ((f * QA + 128) >> 8) + ((f * f * QB + 32768) >> 16);
    const int nn = n < 15 ? n : 15;
    const int r = (2 * g + (1 << nn)) >> (nn + 1);
    return n >= 16 ? 0 : r;
}

extern "C" __global__ void __launch_bounds__(32)
sk18f_union(
    const int* __restrict__ oh,       // [NCH, R, 256]
    const int* __restrict__ ol,
    const int* __restrict__ om,       // [NCH, R]
    const int* __restrict__ os,
    const int* __restrict__ mqb,      // [R]
    const int* __restrict__ dcap,     // [R]
    long long* __restrict__ O,        // [NG, R, 256]
    long long* __restrict__ S,        // [NG, R]
    int R, int NCH, int CPG)
{
    const int r = blockIdx.x;
    const int g = blockIdx.y;
    const int tid = threadIdx.x;
    const int d0 = tid * 8;
    const int mq = mqb[r], dc = dcap[r];
    const int c0 = g * CPG;
    const int c1 = (c0 + CPG) < NCH ? (c0 + CPG) : NCH;

    int mg = -(1 << 30);
    for (int c = 0; c < NCH; ++c) {
        const int mc = om[(size_t)c * R + r];
        mg = mg > mc ? mg : mc;
    }
    long long acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    long long sacc = 0;
    for (int c = c0; c < c1; ++c) {
        int dm = mg - om[(size_t)c * R + r];
        dm = dm < dc ? dm : dc;
        const long long sc = pot2neg_q15((dm * mq + 32768) >> 16);
        const size_t base = ((size_t)c * R + r) * 256 + d0;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const long long oc = (long long)oh[base + e] * 256 + (long long)ol[base + e];
            acc[e] += (oc * sc + 16384) >> 15;
        }
        if (tid == 0) sacc += ((long long)os[(size_t)c * R + r] * sc + 16384) >> 15;
    }
    const size_t ob = ((size_t)g * R + r) * 256 + d0;
#pragma unroll
    for (int e = 0; e < 8; ++e) O[ob + e] = acc[e];
    if (tid == 0) S[(size_t)g * R + r] = sacc;
}
