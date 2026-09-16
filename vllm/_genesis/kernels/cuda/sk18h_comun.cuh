// SPDX-License-Identifier: Apache-2.0
//
// SK-18h — ayudas comunes, TODO ENTERO (ni una instruccion flotante):
//   * fp16 (bits IEEE) -> magnitud fija Q32 en 64 bits (|x| * 2^32, sin desplazamientos
//     negativos: (1024|m) << (e + 7))
//   * fija Q-generica -> bits fp16 con clz y desplazamientos (redondeo al par mas cercano
//     aproximado por medio punto, acarreo al exponente, saturacion a 65504)
//   * mariposa entre lanes (shfl.sync.bfly.b32) para maximos por fila repartidos en hilos
//     (el maximo va en Q16 sin signo: |x| <= 65504 -> < 2^32)
#pragma once

__device__ __forceinline__ unsigned sdir(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ unsigned bfly32(unsigned x, int mask) {
    unsigned y;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=r"(y) : "r"(x), "r"(mask));
    return y;
}
// mariposa con enteros CON SIGNO (misma instruccion, otro tipo)
__device__ __forceinline__ int bfly32s(int x, int mask) {
    int y;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=r"(y) : "r"(x), "r"(mask));
    return y;
}
// maximo sobre los 32 lanes del warp (5 etapas de mariposa)
__device__ __forceinline__ unsigned max_lanes(unsigned x) {
    for (int k = 1; k < 32; k <<= 1) { const unsigned t = bfly32(x, k); x = x > t ? x : t; }
    return x;
}

// |x| en Q32 (64 bits) desde los bits de un fp16; *neg = signo.
__device__ __forceinline__ long long fp16_a_q32(unsigned bits, int* neg) {
    const unsigned e = (bits >> 10) & 31u;
    const unsigned m = bits & 1023u;
    *neg = (int)((bits >> 15) & 1u);
    const unsigned ee = e < 31u ? e : 30u;                  // inf/nan -> maximo finito
    const long long mag = ee ? ((long long)(1024u | m) << (ee + 7)) : ((long long)m << 8);
    return mag;
}

// u (>= 0) * 2^E0 -> bits fp16 (con signo neg). Saturacion a 65504, subflujo a 0.
__device__ __forceinline__ unsigned q_a_fp16(unsigned long long u, int E0, int neg) {
    if (u == 0ull) return (unsigned)neg << 15;
    const int p = 63 - (int)__builtin_clzll(u);              // bit mas alto
    int e = p + E0 + 15;
    long long mant;
    const int sh = p - 10;
    if (sh > 0) mant = (long long)((u + (1ull << (sh - 1))) >> sh);
    else mant = (long long)(u << (-sh));
    if (mant >= 2048) { mant >>= 1; e += 1; }                // acarreo del redondeo
    unsigned r;
    if (e >= 31) r = (30u << 10) | 1023u;
    else if (e <= 0) r = 0u;
    else r = ((unsigned)e << 10) | ((unsigned)mant & 1023u);
    return r | ((unsigned)neg << 15);
}

// 2^-(t/256) en Q15 con 32768 = 1 (t en medios-log Q8), 0 si t >= 16*256.
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

// ── Walsh-Hadamard entero repartido en 32 lanes de 8 dimensiones ─────────────────────────
// x[8] = las 8 dims del lane (Q8 int32, |x| < 2^23 -> la suma de 256 entra en 31 bits).
// Etapas de distancia 1, 2, 4 dentro del lane; 8..128 con mariposa: el lane con el bit
// (dist/8) en 0 tiene j (a) y su pareja j+dist (b) esta en lane ^ (dist/8):
//   bit 0: a + b     bit 1: a - b  = pareja - propio
// Resultado = H * x sin normalizar (Sylvester, H[i][j] = (-1)^popcount(i&j)).
__device__ __forceinline__ void fwht_lanes(int* x, int lane) {
    for (int len = 1; len < 8; len <<= 1)
        for (int i = 0; i < 8; i += 2 * len)
            for (int j = i; j < i + len; ++j) {
                const int a = x[j], b = x[j + len];
                x[j] = a + b; x[j + len] = a - b;
            }
    for (int m = 1; m < 32; m <<= 1) {
        const int alto = (lane & m) != 0;
        for (int e = 0; e < 8; ++e) {
            const int t = (int)bfly32((unsigned)x[e], m);
            x[e] = alto ? (t - x[e]) : (x[e] + t);
        }
    }
}

// fp16 (bits) -> Q8 con signo, recortado a |x| < 2^23 unidades Q8 (32768.0): la suma de las
// 256 dims del Walsh-Hadamard (2^23 * 256 = 2^31) todavia entra en int32.
__device__ __forceinline__ int fp16_a_q8s(unsigned bits) {
    int ng;
    const long long mag = fp16_a_q32(bits, &ng) >> 24;
    const int m = mag > 8388607 ? 8388607 : (int)mag;
    return ng ? -m : m;
}
