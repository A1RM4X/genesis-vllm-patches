// SPDX-License-Identifier: Apache-2.0
//
// SK-18h/union v4 — union de paginas en paralelo por grupos de CPG paginas, con el maximo
// global calculado ADENTRO (antes torch.amax: ~8 us por capa, 19 por paso), solo sobre las
// paginas con tokens de la secuencia; paginas cuyo factor ya es 0 se saltean. Todo entero.
//   mg = max_{c < np} m_c  (lane 0 lo recorre, mariposa lo reparte a los 32 lanes)
//   Og[g, r, d] = sum_{c en grupo g, c < np} (O_c * sc_c + 2^14) >> 15 ;  Sg[g, r] idem con S
// Grilla: x = fila r, y = grupo g; 32 lanes = grupos de 8 dimensiones.
#include <cuda_pipeline.h>
#include "sk18h_comun.cuh"

#define QD 256

extern "C" __global__ void __launch_bounds__(32)
sk18h_union4(
    const int* __restrict__ oh,       // [NCH, R, 256]
    const int* __restrict__ ol,
    const int* __restrict__ om,       // [NCH, R]
    const int* __restrict__ os,
#ifdef O32
    const int* __restrict__ oc,       // [NCH, R] correccion del cero de V (int4)
#endif
    const int* __restrict__ mqb,      // [R]
    const int* __restrict__ dcap,     // [R]
    const int* __restrict__ seq,      // [B]
    long long* __restrict__ Og,       // [NG, R, 256]
    long long* __restrict__ Sg,       // [NG, R]
    int R, int NCH, int CPG, int RB, int BS)
{
    const int r = blockIdx.x;
    const int g = blockIdx.y;
    const int tid = threadIdx.x;
    const int d0 = tid * 8;
    int np = (seq[r / RB] + BS - 1) / BS;
    np = np < NCH ? np : NCH;
    const int c0 = g * CPG;
    int c1 = c0 + CPG;
    c1 = c1 < np ? c1 : np;

    // Maximo global sobre las paginas activas, repartido entre los 32 lanes (antes lo recorria
    // lane 0 en serie: con 69 paginas eran 69 cargas dependientes por bloque).
    unsigned mgu = 0u;                                   // max en orden sin signo (+2^31)
    for (int c = tid; c < np; c += 32) {
        const unsigned u = (unsigned)om[(size_t)c * R + r] ^ 0x80000000u;
        mgu = mgu > u ? mgu : u;
    }
    mgu = max_lanes(mgu);
    const int mg = (int)(mgu ^ 0x80000000u);
    const int mq = mqb[r], dc = dcap[r];

    long long acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    long long sacc = 0;
    for (int c = c0; c < c1; ++c) {
        const int mc = om[(size_t)c * R + r];
        if (mc <= -(1 << 29)) continue;
        int dm = mg - mc;
        dm = dm < dc ? dm : dc;
        const long long sc = pot2neg_q15((dm * mq + 32768) >> 16);
        if (sc == 0) continue;
        const size_t base = ((size_t)c * R + r) * QD + d0;
#ifdef O32
        const long long cc = (long long)oc[(size_t)c * R + r];
#endif
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            long long oc_ = (long long)oh[base + e] * 256 + (long long)ol[base + e];
#ifdef O32   // int4: el cero de V es igual para toda dimension y se resta aca
            oc_ -= cc;
#endif
            acc[e] += (oc_ * sc + 16384) >> 15;
        }
        if (tid == 0) sacc += ((long long)os[(size_t)c * R + r] * sc + 16384) >> 15;
    }
    const size_t ob = ((size_t)g * R + r) * QD + d0;
#pragma unroll
    for (int e = 0; e < 8; ++e) Og[ob + e] = acc[e];
    if (tid == 0) Sg[(size_t)g * R + r] = sacc;
}
