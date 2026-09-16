// SPDX-License-Identifier: Apache-2.0
//
// SK-19/quant — cuantiza la activacion a int8 por token y DEVUELVE LA ESCALA YA MULTIPLICADA
// por el factor global de la capa, en un solo lanzamiento y sin punto flotante.
//
// Por que existe: el camino W4A8 de vLLM hace
//     x_q, a_scales = per_token_quant_int8(x)      <- 560 us por paso (256 lanzamientos)
//     a_scales = a_scales * input_global_scale     <- 272 us por paso (208 lanzamientos)
// y esa segunda linea multiplica M floats (uno por token): es todo latencia de lanzamiento.
//
// Los dos trucos que lo hacen barato, los dos enteros:
//  1. El patron de bits de |x| en fp16 es MONOTONO, asi que el maximo de la fila sale con
//     `bits & 0x7fff` y un max: ni una conversion.
//  2. Para cuantizar no hace falta normalizar a una escala comun de 64 bits. Con la mantisa de
//     11 bits y el exponente de 5 alcanza:  q = (m * rec) >> (DESP - e),  con rec calculado UNA
//     vez por fila. Todo en 32 bits.
//
// Grilla: x = fila (token). Un bloque por fila.
#include "sk18h_comun.cuh"

#ifndef HILOS
#define HILOS 256
#endif
// Trozos de 8 fp16 que cada hilo guarda en registros entre las dos pasadas (sin esto la fila se
// lee dos veces).
#ifndef CHREG
#define CHREG 6
#endif
// q = (m * rec) >> (DESP + e_max - e). m llega a 2^11 y rec = 127*2^DESP/m_max <= 2^DESP/8,
// asi que DESP=23 deja el producto en 2^31 con lugar para el redondeo (con 24 o 26 se desborda).
#define DESP 23

__device__ __forceinline__ int cuantizar_uno(unsigned short b, unsigned rec, int rsh) {
    const int e = (int)((b >> 10) & 31u);
    const unsigned m = (b & 1023u) | (e ? 1024u : 0u);
    int s = rsh - e;                       // desplazamiento variable por elemento
    s = s < 0 ? 0 : (s > 31 ? 31 : s);
    unsigned q = (m * rec + (1u << s >> 1)) >> s;
    q = q < 127u ? q : 127u;
    return (b & 0x8000u) ? -(int)q : (int)q;
}

extern "C" __global__ void __launch_bounds__(HILOS)
sk19_quant(
    const unsigned short* __restrict__ x,    // [M, K] bits fp16
    const unsigned* __restrict__ gbits,      // [1] bits del factor global (fp32)
    signed char* __restrict__ xq,            // [M, K]
    unsigned* __restrict__ esc,              // [M] bits fp32 (escala * factor global)
    int K, int XS)                           // XS = paso de fila de x (elementos)
{
    __shared__ unsigned s_max[HILOS / 32];
    const int fila = blockIdx.x;
    const int tid = threadIdx.x;
    const size_t base = (size_t)fila * XS;

    // Camino vectorizado: 8 fp16 por carga (uint4) y 8 int8 por escritura (uint2).
    const int V8 = ((((base * 2) & 15) == 0) && ((K & 7) == 0) && (((size_t)x & 15) == 0)) ? (K >> 3) : 0;
    const uint4* x4 = (const uint4*)(x + base);
    uint2* q4 = (uint2*)(xq + (size_t)fila * K);
    const int nch = V8 ? ((V8 - tid + HILOS - 1) / HILOS) : 0;
    const int enreg = (nch > 0 && nch <= CHREG);
    uint4 cache[CHREG];

    // 1) maximo de |x|: el patron de bits sin el signo ya ordena los fp16
    unsigned mb = 0;
    if (enreg) {
#pragma unroll
        for (int r = 0; r < CHREG; ++r) {
            if (r >= nch) break;
            const uint4 v = x4[tid + r * HILOS];
            cache[r] = v;
            const unsigned* p = (const unsigned*)&v;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const unsigned a = p[j] & 0x7fff7fffu;
                const unsigned lo = a & 0xffffu, hi = a >> 16;
                mb = mb > lo ? mb : lo;
                mb = mb > hi ? mb : hi;
            }
        }
    } else if (V8) {
        for (int c = tid; c < V8; c += HILOS) {
            const uint4 v = x4[c];
            const unsigned* p = (const unsigned*)&v;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const unsigned a = p[j] & 0x7fff7fffu;
                const unsigned lo = a & 0xffffu, hi = a >> 16;
                mb = mb > lo ? mb : lo;
                mb = mb > hi ? mb : hi;
            }
        }
    } else {
        for (int i = tid; i < K; i += HILOS) {
            const unsigned a = x[base + i] & 0x7fffu;
            mb = mb > a ? mb : a;
        }
    }
    mb = max_lanes(mb);
    if ((tid & 31) == 0) s_max[tid >> 5] = mb;
    __syncthreads();
    if (tid < 32) {
        unsigned v = (tid < HILOS / 32) ? s_max[tid] : 0u;
        v = max_lanes(v);
        if (tid == 0) s_max[0] = v;
    }
    __syncthreads();
    mb = s_max[0];

    // 2) reciproco, UNA vez por fila. Con |max| = mmx * 2^(emx - 25):
    //    q = |x| * 127 / |max| = m_x * 2^(e_x - 25) * 127 / (mmx * 2^(emx - 25))
    //      = (m_x * rec) >> (DESP + emx - e_x),  rec = round(127 * 2^DESP / mmx)
    const int emx = (int)((mb >> 10) & 31u);
    const unsigned mmx = (mb & 1023u) | (emx ? 1024u : 0u);
    const unsigned rec = mmx ? (unsigned)(((127ull << DESP) + (mmx >> 1)) / mmx) : 0u;
    const int rsh = DESP + emx;

    if (enreg) {
#pragma unroll
        for (int r = 0; r < CHREG; ++r) {
            if (r >= nch) break;
            const uint4 v = cache[r];
            const unsigned* p = (const unsigned*)&v;
            unsigned lo = 0, hi = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const unsigned short b = (unsigned short)((p[j >> 1] >> (16 * (j & 1))) & 0xffffu);
                const unsigned byte = (unsigned)(cuantizar_uno(b, rec, rsh) & 0xff);
                if (j < 4) lo |= byte << (8 * j);
                else hi |= byte << (8 * (j - 4));
            }
            uint2 r2; r2.x = lo; r2.y = hi;
            q4[tid + r * HILOS] = r2;
        }
    } else if (V8) {
        for (int c = tid; c < V8; c += HILOS) {
            const uint4 v = x4[c];
            const unsigned* p = (const unsigned*)&v;
            unsigned lo = 0, hi = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const unsigned short b = (unsigned short)((p[j >> 1] >> (16 * (j & 1))) & 0xffffu);
                const unsigned byte = (unsigned)(cuantizar_uno(b, rec, rsh) & 0xff);
                if (j < 4) lo |= byte << (8 * j);
                else hi |= byte << (8 * (j - 4));
            }
            uint2 r2; r2.x = lo; r2.y = hi;
            q4[c] = r2;
        }
    } else {
        for (int i = tid; i < K; i += HILOS)
            xq[(size_t)fila * K + i] = (signed char)cuantizar_uno(x[base + i], rec, rsh);
    }

    // 3) escala = |max| / 127 * global, armada a mano en bits fp32.
    //    |max| = mmx * 2^(emx - 25) para normales; para subnormales, mmx * 2^-24.
    if (tid == 0) {
        int eg;
        const unsigned long long mg = fp32_a_mant(gbits[0], &eg);
        const int e16 = emx ? (emx - 25) : (-24);
        const unsigned long long u = ((unsigned long long)mmx * mg + 63ull) / 127ull;
        esc[fila] = q_a_fp32(u, eg + e16);
    }
}
