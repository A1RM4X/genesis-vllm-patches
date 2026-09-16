// SPDX-License-Identifier: Apache-2.0
//
// SK-18h/final — union de paginas + salida fp16 en UN lanzamiento (antes amax de torch +
// sk18h_union + sk18h_salida: 0,82 ms por paso de decode a 57k en el perfil del stack).
// Recorre SOLO las paginas con tokens de la secuencia (ceil(seq/BS)), aunque la grilla del
// kernel principal sea la del CUDA graph (NCH maximo). Todo entero:
//   mg = max_c m_c ;  sc_c = 2^-(mg - m_c) Q15
//   O = sum_c (O_c * sc_c + 2^14) >> 15 ;  S = sum_c (S_c * sc_c + 2^14) >> 15
//   valor = O * 2^(ev+VSH) / (32767 * S) -> bits fp16
// Grilla: x = token, y = cabeza Q; 32 lanes = grupos de 8 dimensiones.
#include <cuda_pipeline.h>
#include "sk18h_comun.cuh"

#define QD 256

extern "C" __global__ void __launch_bounds__(32)
sk18h_final(
    const int* __restrict__ oh,             // [NCHB, R, 256] (NCHB = paso de la reserva)
    const int* __restrict__ ol,
    const int* __restrict__ om,             // [NCHB, R]
    const int* __restrict__ os,
    const int* __restrict__ mqb,            // [R]
    const int* __restrict__ dcap,           // [R]
    const int* __restrict__ seq,            // [B]
    const int* __restrict__ refs,           // [2] ek, ev
    unsigned short* __restrict__ out,       // [B*L, NH*G, 256] bits fp16
    int NCH, int R, int L, int NH, int G, int MB, int VSH, int BS)
{
    const int t = blockIdx.x;
    const int qh = blockIdx.y;
    const int lane = threadIdx.x;
    const int b = t / L, j = t % L;
    const int kvh = qh / G, g = qh % G;
    const size_t row = ((size_t)b * NH + kvh) * MB + j * G + g;
    const size_t dst = ((size_t)t * NH * G + qh) * QD;
    const int E0 = refs[1] + VSH - 50;
    int np = (seq[b] + BS - 1) / BS;
    np = np < NCH ? np : NCH;

    int mg = -(1 << 30);
    for (int c = 0; c < np; ++c) {
        const int mc = om[(size_t)c * R + row];
        mg = mg > mc ? mg : mc;
    }
    const int mq = mqb[row], dc = dcap[row];
    long long S = 0;
    long long acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int d0 = lane * 8;
    for (int c = 0; c < np; ++c) {
        const int mc = om[(size_t)c * R + row];
        if (mc <= -(1 << 29)) continue;
        int dm = mg - mc;
        dm = dm < dc ? dm : dc;
        const long long sc = pot2neg_q15((dm * mq + 32768) >> 16);
        S += ((long long)os[(size_t)c * R + row] * sc + 16384) >> 15;
        const size_t base = ((size_t)c * R + row) * QD + d0;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const long long oc = (long long)oh[base + e] * 256 + (long long)ol[base + e];
            acc[e] += (oc * sc + 16384) >> 15;
        }
    }
    if (S < 1) S = 1;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const long long O = acc[e];
        const int neg = O < 0;
        const unsigned long long a = (unsigned long long)(neg ? -O : O);
        const unsigned long long r = (a << 20) / (unsigned long long)S;
        out[dst + d0 + e] = (unsigned short)q_a_fp16(r * 32769ull, E0, neg);
    }
}
