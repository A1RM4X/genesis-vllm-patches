// SPDX-License-Identifier: Apache-2.0
//
// SK-18h/prep — queries del decode en UN lanzamiento, todo entero: q fp16 (bits) ->
// Q int8 por token-cabeza en las filas del kernel, lim (causal), mqb y dcap.
//   mqb = kref * sq * 2^ZSH * 2^24 / (32767 * 16 * ln2),  sq = max|q| / 127, kref = 2^ek
//       = (maxQ16 * MQ) >> (56 - (ZSH + 8) - ek),        MQ = round(2^56 / C)
//   dcap = ceil(2^28 / mqb)
// Filas: (b, kvh, j*G + g). Filas >= L*G de cada (b, kvh) quedan con lim = -1.
// Grilla: x = token (b*L + j), y = cabeza Q; 32 lanes = 32 grupos de 8 dimensiones.
#include <cuda_pipeline.h>
#include "sk18h_comun.cuh"

#define QD 256
#ifndef MQ
#define MQ 1561327149LL
#endif

extern "C" __global__ void __launch_bounds__(32)
sk18h_prep(
    const unsigned short* __restrict__ q,   // [B*L, NH*G, 256] bits fp16
    const int* __restrict__ seq,            // [B]
    const int* __restrict__ refs,           // [2] ek, ev
    signed char* __restrict__ Qb,           // [B, NH, MB, 256]
    int* __restrict__ lim,                  // [B, NH, MB]
    int* __restrict__ mqb,
    int* __restrict__ dcap,
    int L, int NH, int G, int MB, int ZSH, int QS)   // QS = paso de fila de q (elementos)
{
    const int t = blockIdx.x;
    const int qh = blockIdx.y;
    const int lane = threadIdx.x;
    const int b = t / L, j = t % L;
    const int kvh = qh / G, g = qh % G;
    const size_t base_bh = ((size_t)b * NH + kvh) * MB;
    const size_t row = base_bh + j * G + g;
    const size_t src = (size_t)t * QS + (size_t)qh * QD;

    unsigned mx = 0;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        int ng;
        const long long a = fp16_a_q32(q[src + lane * 8 + e], &ng) >> 16;
        const unsigned a16 = a > 0xffffffffLL ? 0xffffffffu : (unsigned)a;
        mx = mx > a16 ? mx : a16;
    }
    mx = max_lanes(mx);
    const long long rec = mx ? ((127LL << 24) / (long long)mx) : 0;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int d = lane * 8 + e;
        int ng;
        const long long x = fp16_a_q32(q[src + d], &ng);
        long long v = (x * rec + (1LL << 39)) >> 40;
        v = v < 127 ? v : 127;
        Qb[row * QD + d] = (signed char)(ng ? -v : v);
    }
    if (lane == 0) {
        const int ek = refs[0];
        const int sh = 56 - (ZSH + 8) - ek;
        long long m = sh > 0 ? (((long long)mx * MQ) >> sh) : (((long long)mx * MQ) << (-sh));
        m = m < 1 ? 1 : (m > 2147483647LL ? 2147483647LL : m);
        long long dc = ((1LL << 28) + m - 1) / m;
        dc = dc < (1LL << 30) ? dc : (1LL << 30);
        lim[row] = seq[b] - L + j;
        mqb[row] = (int)m;
        dcap[row] = (int)dc;
        if (j == L - 1 && g == G - 1)
            for (int r = L * G; r < MB; ++r) { lim[base_bh + r] = -1; mqb[base_bh + r] = 1; dcap[base_bh + r] = 0; }
    }
}
