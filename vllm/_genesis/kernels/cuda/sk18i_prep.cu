// SPDX-License-Identifier: Apache-2.0
//
// SK-18i/prep — como sk18h_prep2 pero cuantizando q a NIBBLES por grupos de 64 dims:
//   q~ = H (s * q) / 16 (Walsh-Hadamard entero), luego por grupo g de 64 dims
//       escala_g = qmax * rq_g / 255,  nibble = round(q / escala_g * 7) en [-7, 7]
// Salidas: Qb [B, NH, MB, 128] (dim par en el nibble bajo), rq [B, NH, MB, 4] (uint8 en int32),
// lim (causal), mqb y dcap. El producto q.k del kernel suma por grupo:
//   z = (sum_g acc_g * rq_g * rk_g * kmax) >> ZSH4
// y mqb lleva el resto de la constante (qmax/7, 255^2, 2^ZSH4, 2^ek):
//   mqb = qmax_Q8 * MQ4 >> (56 - ZSH4 - ek - 16),  MQ4 = 2^56 / (7 * 32767 * 255^2 * 16 * ln2)
// Grilla: x = token (b*L + j), y = cabeza Q; 32 lanes = 32 grupos de 8 dimensiones.
#include <cuda_pipeline.h>
#include "sk18h_comun.cuh"

#define QD 256
#ifndef MQ4
#define MQ4 435660LL
#endif
// 1 plano: q en nibbles [-7, 7]. 2 planos: q int8 [-127, 127] partido en dos nibbles con signo
// (q = 16 * alto + bajo), con la misma escala de grupo: el kernel suma 16 * acc_alto + acc_bajo.
#ifndef QPLANOS
#define QPLANOS 1
#endif
// Con 2 planos el tope es 119 = 7*16 + 7: q = 127 daria alto = 8, fuera del rango s4 [-8, 7]
// (se guardaria como -8 y el logit salia con el signo cambiado).
#ifndef NIV
#define NIV (QPLANOS == 2 ? 119 : 7)
#endif

__device__ __forceinline__ unsigned max_xor8(unsigned x) {
    for (int k = 1; k < 8; k <<= 1) { const unsigned t = bfly32(x, k); x = x > t ? x : t; }
    return x;
}

extern "C" __global__ void __launch_bounds__(32)
sk18i_prep(
    const unsigned short* __restrict__ q,   // [B*L, NH*G, 256] bits fp16
    const int* __restrict__ seq,            // [B]
    const int* __restrict__ refs,           // [2] ek, ev
    const int* __restrict__ signos,         // [256] +-1
    unsigned char* __restrict__ Qb,         // [B, NH, MB, QPLANOS*128]
    int* __restrict__ rq,                   // [B, NH, MB, 4]
    int* __restrict__ sq,                   // [B, NH, MB, QPLANOS, 4]: rq_g * suma de nibbles
    int* __restrict__ lim,                  // [B, NH, MB]
    int* __restrict__ mqb,
    int* __restrict__ dcap,
    int L, int NH, int G, int MB, int ZSH4, int QS)
{
    const int t = blockIdx.x;
    const int qh = blockIdx.y;
    const int lane = threadIdx.x;
    const int b = t / L, j = t % L;
    const int kvh = qh / G, g = qh % G;
    const size_t base_bh = ((size_t)b * NH + kvh) * MB;
    const size_t row = base_bh + j * G + g;
    const size_t src = (size_t)t * QS + (size_t)qh * QD;

    int x[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int d = lane * 8 + e;
        x[e] = fp16_a_q8s(q[src + d]) * signos[d];
    }
    fwht_lanes(x, lane);
    unsigned mx = 0;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        x[e] = x[e] >> 4;
        const unsigned a = (unsigned)(x[e] < 0 ? -x[e] : x[e]);
        mx = mx > a ? mx : a;
    }
    const unsigned mg = max_xor8(mx);                    // maximo del grupo de 64 dims
    mx = max_lanes(mg);                                  // maximo de la fila (Q8)
    const int r = mx ? (int)(((long long)mg * 255 + (long long)mx - 1) / (long long)mx) : 0;
    const long long mef = ((long long)mx * r + 254) / 255;
    const long long rec = mef ? (((long long)NIV << 24) / mef) : 0;
    unsigned char bq[4], bh[4];
    int sbajo = 0, salto = 0;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const long long a = x[e] < 0 ? -(long long)x[e] : (long long)x[e];
        long long v = (a * rec + (1LL << 23)) >> 24;
        v = v < NIV ? v : NIV;
        const int q = (int)(x[e] < 0 ? -v : v);
        // QPLANOS=2: q int8 = 16 * alto + bajo, los dos nibbles con signo en [-8, 7]
        const int bajo = QPLANOS == 2 ? (((q + 8) & 15) - 8) : q;
        const int alto = QPLANOS == 2 ? ((q - bajo) >> 4) : 0;
        sbajo += bajo; salto += alto;
        if ((e & 1) == 0) { bq[e >> 1] = (unsigned char)(bajo & 15); bh[e >> 1] = (unsigned char)(alto & 15); }
        else { bq[e >> 1] = (unsigned char)(bq[e >> 1] | ((bajo & 15) << 4));
               bh[e >> 1] = (unsigned char)(bh[e >> 1] | ((alto & 15) << 4)); }
    }
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
        Qb[row * QPLANOS * (QD / 2) + lane * 4 + jj] = bq[jj];
        if (QPLANOS == 2) Qb[(row * QPLANOS + 1) * (QD / 2) + lane * 4 + jj] = bh[jj];
    }
    // sumas del grupo (64 dims = 8 lanes) para el termino del cero por grupo de K
    for (int k = 1; k < 8; k <<= 1) { sbajo += bfly32s(sbajo, k); salto += bfly32s(salto, k); }
    if ((lane & 7) == 0) {
        const int g = lane >> 3;
        rq[row * 4 + g] = r;
        sq[row * QPLANOS * 4 + g] = r * sbajo;
        if (QPLANOS == 2) sq[(row * QPLANOS + 1) * 4 + g] = r * salto;
    }
    if (lane == 0) {
        const int ek = refs[0];
        const int sh = 56 - ZSH4 - ek - 16;
        long long m = sh > 0 ? (((long long)mx * MQ4) >> sh) : (((long long)mx * MQ4) << (-sh));
        m = m < 1 ? 1 : (m > 2147483647LL ? 2147483647LL : m);
        long long dc = ((1LL << 28) + m - 1) / m;
        dc = dc < (1LL << 30) ? dc : (1LL << 30);
        lim[row] = seq[b] - L + j;
        mqb[row] = (int)m;
        dcap[row] = (int)dc;
        if (j == L - 1 && g == G - 1)
            for (int rr = L * G; rr < MB; ++rr) {
                lim[base_bh + rr] = -1; mqb[base_bh + rr] = 1; dcap[base_bh + rr] = 0;
                for (int g = 0; g < 4; ++g) {
                    rq[(base_bh + rr) * 4 + g] = 0;
                    for (int pl = 0; pl < QPLANOS; ++pl) sq[((base_bh + rr) * QPLANOS + pl) * 4 + g] = 0;
                }
            }
    }
}
