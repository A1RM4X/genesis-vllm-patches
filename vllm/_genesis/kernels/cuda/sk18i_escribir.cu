// SPDX-License-Identifier: Apache-2.0
//
// SK-18i/escribir — escritura de la KV INT4 de PN131 dentro de la reserva de vLLM
// int4_per_token_head (264 B por token-cabeza), TODO ENTERO y ASIMETRICA (cero por grupo):
//   [0, KOFF)      K nibbles  [BS][NH][128]     (dim par en el nibble bajo)
//   [KOFF, EOFF)   V nibbles  [NH][256][BS/2]   (token par en el nibble bajo)
//   [EOFF, ...)    escalas    [BS][NH][8]:
//        [0,1]  KM 12 bits (escala mayor de K) | vzp+8 en los 4 bits altos
//        [2..5] por grupo de 64 dims: (r-1) en el nibble bajo, zp+8 en el alto
//        [6,7]  svf int16 (escala de V)
// Valores: k = (nib - zp_g) * KM * r_g / 16 * 2^ek / 32767 ;  v = (nib - vzp) * svf * 2^ev / 32767
// La escala sale de (max - min) / 15 con RECORTE (CLIP/256 alrededor de la media), no del maximo.
// k y v van rotados con Hadamard entero (mismos signos que q).
//
// Dos tokens vecinos comparten byte en V: se lanza DOS VECES, PAR = 0 y 1, y cada lanzamiento
// toma solo los tokens cuyo slot tiene esa paridad (sin lectura-modificacion-escritura cruzada).
// Grilla: x = token, y = cabeza KV; 32 lanes = 32 grupos de 8 dimensiones.
#include <cuda_pipeline.h>
#include "sk18h_comun.cuh"

#define QD 256
// round(32767 * 2^16 / 15): escala (max-min)/15 -> entero relativo a 2^e
#define MESCK 143168102LL
// recorte de la escala: (x - mu) * CLIP / 256 (243/256 = 0,949, el mejor del barrido)
#ifndef CLIP
#define CLIP 243
#endif

__device__ __forceinline__ int min_xor(int x, int hasta) {
    for (int k = 1; k < hasta; k <<= 1) { const int t = bfly32s(x, k); x = x < t ? x : t; }
    return x;
}
__device__ __forceinline__ int max_xor(int x, int hasta) {
    for (int k = 1; k < hasta; k <<= 1) { const int t = bfly32s(x, k); x = x > t ? x : t; }
    return x;
}
__device__ __forceinline__ int sum_xor(int x, int hasta) {
    for (int k = 1; k < hasta; k <<= 1) x += bfly32s(x, k);
    return x;
}

// recorte alrededor de la media: devuelve lo' y hi' (Q8)
__device__ __forceinline__ void recortar(int lo, int hi, int mu, int* lo2, int* hi2) {
    *lo2 = mu + (int)(((long long)(lo - mu) * CLIP) >> 8);
    *hi2 = mu + (int)(((long long)(hi - mu) * CLIP) >> 8);
}

extern "C" __global__ void __launch_bounds__(32)
sk18i_escribir(
    const unsigned short* __restrict__ key,     // [n, NH, 256] bits fp16
    const unsigned short* __restrict__ val,
    const long long* __restrict__ slot,         // [n]
    unsigned char* __restrict__ pool,
    const int* __restrict__ refs,               // [2] ek, ev
    const int* __restrict__ signos,             // [256] +-1
    int NH, int BS, int BLK, int VSH, int KS, int VS, int PAR)
{
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int lane = threadIdx.x;
    const long long sl = slot[t];
    if (sl < 0) return;
    const int o = (int)(sl % BS);
    if ((o & 1) != PAR) return;
    const int ek = refs[0], ev = refs[1];

    const size_t fk = (size_t)t * KS + (size_t)h * QD;
    const size_t fv = (size_t)t * VS + (size_t)h * QD;
    int xk[8], xv[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int d = lane * 8 + e;
        xk[e] = fp16_a_q8s(key[fk + d]) * signos[d];
        xv[e] = fp16_a_q8s(val[fv + d]) * signos[d];
    }
    fwht_lanes(xk, lane);
    fwht_lanes(xv, lane);
    int lok = 1 << 30, hik = -(1 << 30), lov = 1 << 30, hiv = -(1 << 30);
    int sk_ = 0, sv_ = 0;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        xk[e] >>= 4; xv[e] >>= 4;                       // / sqrt(256)
        lok = lok < xk[e] ? lok : xk[e]; hik = hik > xk[e] ? hik : xk[e];
        lov = lov < xv[e] ? lov : xv[e]; hiv = hiv > xv[e] ? hiv : xv[e];
        sk_ += xk[e]; sv_ += xv[e];
    }
    // K: minimo, maximo y media del GRUPO de 64 dims (8 lanes); V: de las 256 dims (32 lanes)
    const int lkg = min_xor(lok, 8), hkg = max_xor(hik, 8);
    const int muk = sum_xor(sk_, 8) >> 6;
    int lk2, hk2; recortar(lkg, hkg, muk, &lk2, &hk2);
    const int Dg = hk2 > lk2 ? (hk2 - lk2) : 0;                  // rango del grupo (Q8)
    const int Dmax = max_xor(Dg, 32);
    const int lv = min_xor(lov, 32), hv = max_xor(hiv, 32);
    const int muv = sum_xor(sv_, 32) >> 8;
    int lv2, hv2; recortar(lv, hv, muv, &lv2, &hv2);
    const int Dv = hv2 > lv2 ? (hv2 - lv2) : 0;

    // escalas guardadas
    long long KM = ((long long)Dmax * MESCK + (1LL << (23 + ek))) >> (24 + ek);
    KM = KM < 1 ? 1 : (KM > 4095 ? 4095 : KM);
    long long svf = ((long long)Dv * MESCK + (1LL << (23 + ev))) >> (24 + ev);
    svf = svf < 1 ? 1 : (svf > (1LL << VSH) ? (1LL << VSH) : svf);
    const int r = Dmax ? (int)(((long long)Dg * 16 + Dmax - 1) / Dmax) : 1;
    const int rg = r < 1 ? 1 : (r > 16 ? 16 : r);

    // reciprocos de las escalas EFECTIVAS (las que va a usar el kernel al decuantizar)
    //   s_k(Q8) = KM * rg * 2^(ek+8) / (16 * 32767)   ->  x / s = (x * reck) >> (24 + ek + 8)
    //   s_v(Q8) = svf * 2^(ev+8) / 32767              ->  x / s = (x * recv) >> (24 + ev + 8)
    const long long reck = (((16LL * 32767) << 24) + (KM * rg) / 2) / (KM * rg);
    const long long recv = (((long long)32767 << 24) + svf / 2) / svf;
    const int shk = 32 + ek, shv = 32 + ev;
    // cero por grupo: nib = round(x / s) + zp, con zp tal que el minimo caiga en -8
    const long long qlo_k = ((long long)lk2 * reck + (1LL << (shk - 1))) >> shk;
    const long long qlo_v = ((long long)lv2 * recv + (1LL << (shv - 1))) >> shv;
    int zpk = (int)(-8 - qlo_k); zpk = zpk < -8 ? -8 : (zpk > 7 ? 7 : zpk);
    int zpv = (int)(-8 - qlo_v); zpv = zpv < -8 ? -8 : (zpv > 7 ? 7 : zpv);

    const size_t base = (size_t)(sl / BS) * (size_t)BLK;
    const size_t KOFF = (size_t)BS * NH * (QD / 2);
    const size_t EOFF = 2 * KOFF;
    unsigned char bk[4];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int d = lane * 8 + e;
        long long qk = (((long long)xk[e] * reck + (1LL << (shk - 1))) >> shk) + zpk;
        long long qv = (((long long)xv[e] * recv + (1LL << (shv - 1))) >> shv) + zpv;
        qk = qk < -8 ? -8 : (qk > 7 ? 7 : qk);
        qv = qv < -8 ? -8 : (qv > 7 ? 7 : qv);
        const int nk = (int)qk & 15, nv = (int)qv & 15;
        if ((e & 1) == 0) bk[e >> 1] = (unsigned char)nk;
        else bk[e >> 1] = (unsigned char)(bk[e >> 1] | (nk << 4));
        const size_t pv = base + KOFF + ((size_t)h * QD + d) * (BS / 2) + (o >> 1);
        const unsigned char viejo = pool[pv];
        pool[pv] = (o & 1) ? (unsigned char)((viejo & 0x0f) | (nv << 4))
                           : (unsigned char)((viejo & 0xf0) | nv);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
        pool[base + ((size_t)o * NH + h) * (QD / 2) + lane * 4 + j] = bk[j];
    if ((lane & 7) == 0)                                  // grupo g = lane >> 3
        pool[base + EOFF + ((size_t)o * NH + h) * 8 + 2 + (lane >> 3)] =
            (unsigned char)((rg - 1) | ((zpk + 8) << 4));
    if (lane == 0) {
        const size_t pe = base + EOFF + ((size_t)o * NH + h) * 8;
        const unsigned w = (unsigned)KM | ((unsigned)(zpv + 8) << 12);
        pool[pe] = (unsigned char)(w & 255);
        pool[pe + 1] = (unsigned char)((w >> 8) & 255);
        pool[pe + 6] = (unsigned char)(svf & 255);
        pool[pe + 7] = (unsigned char)((svf >> 8) & 255);
    }
}
