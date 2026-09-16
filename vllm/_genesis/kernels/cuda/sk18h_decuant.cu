// SPDX-License-Identifier: Apache-2.0
//
// SK-18h/decuant — paginas de la KV de PN131 -> cache fp16 [n, 2, BS, NH, 256] para el prefill
// (FlashInfer paginado), TODO ENTERO: valor = x8 * sf * 2^e / 32767 -> bits fp16 con clz.
//   u = |x8| * sf * 32769  (~ |x8| * sf * 2^30 / 32767),  valor = u * 2^(e - 30)
// Grilla: x = bloque de la lista, y = token de la pagina; 32 lanes = grupos de 8 dimensiones.
#include <cuda_pipeline.h>
#include "sk18h_comun.cuh"

#define QD 256
// ROTV=1: la V del pool viene ROTADA (es el espejo int8 del camino int4) y hay que des-rotarla
// con el Walsh-Hadamard entero, como hace sk18i_decuant.
#ifndef ROTV
#define ROTV 0
#endif

extern "C" __global__ void __launch_bounds__(32)
sk18h_decuant(
    const signed char* __restrict__ pool,
    const long long* __restrict__ ids,       // [n] bloques fisicos
    const int* __restrict__ refs,            // [2] ek, ev
    const int* __restrict__ signos,          // [256] +-1 (solo con ROTV)
    unsigned short* __restrict__ out,        // [n, 2, BS, NH, 256] bits fp16
    int BLK, int BS, int NH)
{
    const int i = blockIdx.x;
    const int o = blockIdx.y;
    const int lane = threadIdx.x;
    if (ids[i] < 0) return;                  // esta pagina la hace el otro kernel
    const size_t base = (size_t)ids[i] * (size_t)BLK;
    const size_t KOFF = (size_t)BS * NH * QD;
    const int ek = refs[0], ev = refs[1];
#pragma unroll
    for (int h = 0; h < NH; ++h) {
        const unsigned char* e8 = (const unsigned char*)pool + base + 2 * KOFF + ((size_t)o * NH + h) * 4;
        const int skf = (int)(short)(e8[0] | (e8[1] << 8));
        const int svf = (int)(short)(e8[2] | (e8[3] << 8));
        const size_t ok_ = (((size_t)i * 2 + 0) * BS + o) * NH * QD + (size_t)h * QD;
        const size_t ov_ = (((size_t)i * 2 + 1) * BS + o) * NH * QD + (size_t)h * QD;
        int xv[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int d = lane * 8 + e;
            const int kx = pool[base + ((size_t)o * NH + h) * QD + d];
            const int vx = pool[base + KOFF + ((size_t)h * QD + d) * BS + o];
            const unsigned long long uk = (unsigned long long)(kx < 0 ? -kx : kx) * (unsigned long long)skf * 32769ull;
            out[ok_ + d] = (unsigned short)q_a_fp16(uk, ek - 30, kx < 0);
            if (ROTV) {
                xv[e] = vx * (int)svf;
            } else {
                const unsigned long long uv = (unsigned long long)(vx < 0 ? -vx : vx) * (unsigned long long)svf * 32769ull;
                out[ov_ + d] = (unsigned short)q_a_fp16(uv, ev - 30, vx < 0);
            }
        }
        if (ROTV) {
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
}
