// SPDX-License-Identifier: Apache-2.0
//
// SK-18h/escribir — escritura de la KV de PN131 en UN lanzamiento, todo entero.
// Entradas fp16 como bits. Por (token, cabeza): maximo |x| con mariposa entre lanes,
// reciproco una vez, int8 por dimension, escala int16 relativa a la referencia 2^ek.
// Layout del bloque: K [BS][NH][256] | V [NH][256][BS] | escalas int16 [BS][NH][2].
// Grilla: x = token, y = cabeza KV; 32 lanes = 32 grupos de 8 dimensiones.
#include <cuda_pipeline.h>
#include "sk18h_comun.cuh"

#define QD 256
// round(32767 / 127 * 2^16)
#define MESC 16908804LL

__device__ __forceinline__ unsigned q16(long long a) {
    const long long b = a >> 16;
    return b > 0xffffffffLL ? 0xffffffffu : (unsigned)b;
}

extern "C" __global__ void __launch_bounds__(32)
sk18h_escribir(
    const unsigned short* __restrict__ key,     // [n, NH, 256] bits fp16
    const unsigned short* __restrict__ val,
    const long long* __restrict__ slot,         // [n]
    signed char* __restrict__ pool,
    const int* __restrict__ refs,               // [2] ek, ev (referencias 2^ek, 2^ev)
    int NH, int BS, int BLK, int VSH, int KS, int VS)   // pasos de fila de k y v
{
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int lane = threadIdx.x;
    const size_t fila = (size_t)t * KS + (size_t)h * QD;
    const size_t filav = (size_t)t * VS + (size_t)h * QD;

    unsigned mk = 0, mv = 0;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        int ng;
        unsigned a = q16(fp16_a_q32(key[fila + lane * 8 + e], &ng));
        unsigned b = q16(fp16_a_q32(val[filav + lane * 8 + e], &ng));
        mk = mk > a ? mk : a;
        mv = mv > b ? mv : b;
    }
    mk = max_lanes(mk);
    mv = max_lanes(mv);

    const long long sl = slot[t];
    if (sl < 0) return;
    const int ek = refs[0], ev = refs[1];
    const size_t base = (size_t)(sl / BS) * (size_t)BLK;
    const int o = (int)(sl % BS);
    const size_t KOFF = (size_t)BS * NH * QD;
    const long long reck = mk ? ((127LL << 24) / (long long)mk) : 0;
    const long long recv = mv ? ((127LL << 24) / (long long)mv) : 0;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int d = lane * 8 + e;
        int nk, nv;
        const long long xk = fp16_a_q32(key[fila + d], &nk);
        const long long xv = fp16_a_q32(val[filav + d], &nv);
        long long qk = (xk * reck + (1LL << 39)) >> 40;
        long long qv = (xv * recv + (1LL << 39)) >> 40;
        qk = qk < 127 ? qk : 127;
        qv = qv < 127 ? qv : 127;
        pool[base + ((size_t)o * NH + h) * QD + d] = (signed char)(nk ? -qk : qk);
        pool[base + KOFF + ((size_t)h * QD + d) * BS + o] = (signed char)(nv ? -qv : qv);
    }
    if (lane == 0) {
        long long skf = ((long long)mk * MESC + (1LL << (31 + ek))) >> (32 + ek);
        long long svf = ((long long)mv * MESC + (1LL << (31 + ev))) >> (32 + ev);
        skf = skf < 32767 ? skf : 32767;
        svf = svf < (1LL << VSH) ? svf : (1LL << VSH);
        const size_t pe = base + 2 * KOFF + ((size_t)o * NH + h) * 4;
        pool[pe] = (signed char)(skf & 255);
        pool[pe + 1] = (signed char)((skf >> 8) & 255);
        pool[pe + 2] = (signed char)(svf & 255);
        pool[pe + 3] = (signed char)((svf >> 8) & 255);
    }
}
