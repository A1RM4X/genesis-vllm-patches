// SPDX-License-Identifier: Apache-2.0
//
// SK-18i/salida — union de tramos -> salida fp16, con la DES-ROTACION de V adentro (en int4 la
// V se guarda rotada con Hadamard, asi que la salida vive en el dominio rotado):
//   r_d = O_d * 256 / S           (entero con signo, |r| <= 7 * 2^VSH * 256)
//   y   = H r                     (Walsh-Hadamard entero en lanes; |y| <= 256 |r| < 2^31)
//   v_d = s_d * y_d * 2^(ev + VSH - 8 - 4) / 32767
// u = |y| * 32769 (~ |y| * 2^30 / 32767) -> bits fp16 con clz, E0 = ev + VSH - 42.
// Grilla: x = token, y = cabeza Q; 32 lanes = 32 grupos de 8 dimensiones.
#include <cuda_pipeline.h>
#include "sk18h_comun.cuh"

#define QD 256

extern "C" __global__ void __launch_bounds__(32)
sk18i_salida(
    const long long* __restrict__ Og,       // [NG, R, 256]
    const long long* __restrict__ Sg,       // [NG, R]
    const int* __restrict__ refs,           // [2] ek, ev
    const int* __restrict__ signos,         // [256] +-1
    unsigned short* __restrict__ out,       // [B*L, NH*G, 256] bits fp16
    int NG, int R, int L, int NH, int G, int MB, int VSH)
{
    const int t = blockIdx.x;
    const int qh = blockIdx.y;
    const int lane = threadIdx.x;
    const int b = t / L, j = t % L;
    const int kvh = qh / G, g = qh % G;
    const size_t row = ((size_t)b * NH + kvh) * MB + j * G + g;
    const size_t dst = ((size_t)t * NH * G + qh) * QD;
    const int E0 = refs[1] + VSH - 42;

    long long S = 0;
    for (int k = 0; k < NG; ++k) S += Sg[(size_t)k * R + row];
    if (S < 1) S = 1;
    int x[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int d = lane * 8 + e;
        long long O = 0;
        for (int k = 0; k < NG; ++k) O += Og[((size_t)k * R + row) * QD + d];
        const int neg = O < 0;
        const unsigned long long a = (unsigned long long)(neg ? -O : O);
        const long long r = (long long)(((a << 8) + (unsigned long long)(S >> 1)) / (unsigned long long)S);
        x[e] = (int)(neg ? -r : r);
    }
    fwht_lanes(x, lane);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int d = lane * 8 + e;
        const int y = x[e] * signos[d];
        const int neg = y < 0;
        const unsigned long long u = (unsigned long long)(neg ? -(long long)y : (long long)y) * 32769ull;
        out[dst + d] = (unsigned short)q_a_fp16(u, E0, neg);
    }
}
