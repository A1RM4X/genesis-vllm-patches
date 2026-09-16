// SPDX-License-Identifier: Apache-2.0
//
// SK-18i/decuant — paginas de la KV INT4 -> cache fp16 [n, 2, BS, NH, 256] para el prefill,
// TODO ENTERO. K sale en el dominio ROTADO (la q del prefill tambien se rota); V sale
// DES-ROTADA (Walsh-Hadamard entero + signos), que es lo que consume FlashInfer.
//   k_d = (nib - zp_g) * KM * r_g / 16 * 2^ek / 32767  u = |A| * 524304, E = ek - 38
//   v   = s * H((nib - vzp) * svf) / 16 * 2^ev / 32767 u = |y| * 32769,          E = ev - 34
// Grilla: x = bloque de la lista, y = token de la pagina; 32 lanes = grupos de 8 dimensiones.
#include <cuda_pipeline.h>
#include "sk18h_comun.cuh"

#define QD 256
// round(2^38 / (16 * 32767))
#define RK16 524304LL

__device__ __forceinline__ int nib(unsigned char b, int alto) {
    const int n = alto ? (b >> 4) : (b & 15);
    return n >= 8 ? n - 16 : n;
}

extern "C" __global__ void __launch_bounds__(32)
sk18i_decuant(
    const unsigned char* __restrict__ pool,
    const long long* __restrict__ ids,       // [n] bloques fisicos
    const int* __restrict__ refs,            // [2] ek, ev
    const int* __restrict__ signos,          // [256] +-1
    unsigned short* __restrict__ out,        // [n, 2, BS, NH, 256] bits fp16
    int BLK, int BS, int NH)
{
    const int i = blockIdx.x;
    const int o = blockIdx.y;
    const int lane = threadIdx.x;
    const size_t base = (size_t)ids[i] * (size_t)BLK;
    const size_t KOFF = (size_t)BS * NH * (QD / 2);
    const int ek = refs[0], ev = refs[1];
    for (int h = 0; h < NH; ++h) {
        const unsigned char* pe = pool + base + 2 * KOFF + ((size_t)o * NH + h) * 8;
        const long long kmax = (long long)((pe[0] | (pe[1] << 8)) & 4095);
        const int vzp = (int)((pe[1] >> 4) & 15) - 8;
        const long long svf = (long long)(pe[6] | (pe[7] << 8));
        const size_t ok_ = (((size_t)i * 2 + 0) * BS + o) * NH * QD + (size_t)h * QD;
        const size_t ov_ = (((size_t)i * 2 + 1) * BS + o) * NH * QD + (size_t)h * QD;
        int xv[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int d = lane * 8 + e;
            const unsigned char bk = pool[base + ((size_t)o * NH + h) * (QD / 2) + (d >> 1)];
            const int b = (int)pe[2 + (d >> 6)];
            const long long rk = (b & 15) + 1, zp = ((b >> 4) & 15) - 8;
            const long long A = (long long)(nib(bk, d & 1) - zp) * kmax * rk;
            const unsigned long long uk = (unsigned long long)(A < 0 ? -A : A) * (unsigned long long)RK16;
            out[ok_ + d] = (unsigned short)q_a_fp16(uk, ek - 38, A < 0);
            const unsigned char bv = pool[base + KOFF + ((size_t)h * QD + d) * (BS / 2) + (o >> 1)];
            xv[e] = (nib(bv, o & 1) - vzp) * (int)svf;
        }
        fwht_lanes(xv, lane);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int d = lane * 8 + e;
            const int y = xv[e] * signos[d];
            const unsigned long long uv = (unsigned long long)(y < 0 ? -(long long)y : (long long)y) * 32769ull;
            out[ov_ + d] = (unsigned short)q_a_fp16(uv, ev - 34, y < 0);
        }
    }
}
