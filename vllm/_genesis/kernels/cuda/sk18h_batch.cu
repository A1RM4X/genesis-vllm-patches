// SPDX-License-Identifier: Apache-2.0
//
// SK-18h — SK-18g para el backend de vLLM (PN131): lote de secuencias en un lanzamiento,
// KV dentro de la reserva de int8_per_token_head (520 B por token-cabeza, layout propio),
// longitudes y tabla de bloques leidas en la GPU (sin sincronizar con el host).
//
// Grilla: x = pagina logica p (NCH = ceil(max_seq_len / BS)), y = secuencia x cabeza KV.
// Filas de query: MB = 32 por (secuencia, cabeza) = hasta 5 tokens x 6 cabezas Q.
//
// Bloque fisico (BLK bytes = stride del bloque en el tensor de vLLM):
//   [0, KOFF)      K int8 [BS][NH][256]
//   [KOFF, EOFF)   V int8 [NH][256][BS]
//   [EOFF, ...)    escalas int16 [BS][NH][2] (skf, svf) relativas a la referencia de la capa
// Escalas con referencia holgada (64x el maximo inicial) sin perder bits:
//   z' = (acc * skf) >> ZSH  en 64 bits;  w' = (w * svf + 2^(VSH-1)) >> VSH.
#include <cuda_fp16.h>
#include <cuda_pipeline.h>

#define QD 256          // dims de q/k/v
#define NQW 8           // queries por warp
#ifndef NWARPS
#define NWARPS 4        // warps por bloque -> 32 queries por bloque
#endif
#define BQ (NQW * NWARPS)
#define BK 64           // keys por unidad
#define MT (QD / 16)    // fragmentos de 16 dims (w.v)
#define CHQ (QD / 16)
#define CHV (BK / 16)
#define PADZ (-(1 << 28))
// DIAG (perfil por seccion, NO para produccion): 1 sin mma de Q.K, 2 sin reescalado,
// 3 sin mma de w.v, 4 sin carga de V, 5 sin carga de K, 6 sin guardar pesos, 7 sin la
// barrera antes de w.v, 8 sin calculo de pesos (w = 0 constante).
#ifndef DIAG
#define DIAG 0
#endif
// SALTEAR=1: reescalado y planos de peso nulos se saltean por warp (resultado identico).
#ifndef SINCRONO
#define SINCRONO 1
#endif
#ifndef SALTEAR
#define SALTEAR 1
#endif
// ESPERA: grupos de cp.async a esperar antes de leer la unidad. 0 = esperar todo.
// 1 (pipeline de 2 etapas como SK-12) tiene una CARRERA: a 20-22k tokens lee a veces
// fragmentos de V^T sin terminar de copiar (hi/lo distintos del emulador en tramos y
// warps al azar; 6 corridas de 6 con error). ESPERA=0 es exacto siempre y cuesta ~6%.
#ifndef ESPERA
#define ESPERA 0
#endif
#define NHMAX 2         // cabezas KV por placa (TP=2)
#define MINIT (-(1 << 30))

#ifndef QA
#define QA 22474
#endif
#ifndef QB
#define QB 6090
#endif

__device__ __forceinline__ int swz(int row, int byte_col, int ch) {
    return ((byte_col >> 4) ^ (row & (ch - 1))) << 4;
}
__device__ __forceinline__ int swzB(int row, int mat, int byte_col, int ch) {
    const int chunk = (mat * ch) | (byte_col >> 4);
    return (chunk ^ (row & (ch - 1))) << 4;
}
__device__ __forceinline__ unsigned sdir(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ int pot2neg_q15(int t) {
    const int n = t >> 8, f = t & 255;
    const int g = 32768 - ((f * QA + 128) >> 8) + ((f * f * QB + 32768) >> 16);
    const int nn = n < 15 ? n : 15;
    const int r = (2 * g + (1 << nn)) >> (nn + 1);
    return n >= 16 ? 0 : r;
}
__device__ __forceinline__ int bfly(int x, int mask) {
    int y;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=r"(y) : "r"(x), "r"(mask));
    return y;
}

// OR/max de un entero chico sobre los 32 lanes del warp (5 etapas de mariposa)
__device__ __forceinline__ int any_lanes(int x) {
    for (int k = 1; k < 32; k <<= 1) { const int t = bfly(x, k); x = x > t ? x : t; }
    return x;
}

extern "C" __global__ void __launch_bounds__(NWARPS * 32)
sk18h_batch(
    const signed char* __restrict__ Q,      // [B, NH, 32, 256]
    const signed char* __restrict__ pool,   // bloques fisicos contiguos de BLK bytes
    const int* __restrict__ bt,             // [B, BTS] tabla de bloques
    const int* __restrict__ nseq,           // [B] tokens en KV por secuencia
    const int* __restrict__ lim,            // [B, NH, 32] ultima key visible (-1 = fila vacia)
    const int* __restrict__ mqb,            // [B, NH, 32]
    const int* __restrict__ dcap,           // [B, NH, 32]
    int* __restrict__ out_hi,               // [NCH, R, 256]  R = B*NH*32
    int* __restrict__ out_lo,
    int* __restrict__ out_m,                // [NCH, R]
    int* __restrict__ out_s,
    int BLK, int BTS, int CHK, int NCH, int NH, int ZSH, int VSH)   // CHK = BS; BLK = bytes por bloque
{
    // Shared: Q^T [BQ][256] | K [2][64][256] | V^T [2][256][64] | W [BQ][128]
    extern __shared__ signed char _smem[];
    signed char (*sQ)[QD]         = (signed char (*)[QD]) _smem;
    signed char (*sK)[BK][QD]     = (signed char (*)[BK][QD]) (_smem + BQ * QD);
    signed char (*sV)[QD][BK]     = (signed char (*)[QD][BK]) (_smem + BQ * QD + 2 * BK * QD);
    signed char (*sW)[2 * BK]     = (signed char (*)[2 * BK]) (_smem + BQ * QD + 2 * BK * QD + 2 * QD * BK);
    // Escalas de la unidad (skf, svf) en shared: antes eran cargas globales por key
    // (long_scoreboard 0,93 contra 0,23 de FlashInfer en ncu).
    signed char (*sE)[BK * NHMAX * 4] = (signed char (*)[BK * NHMAX * 4]) (_smem + BQ * QD + 2 * BK * QD + 2 * QD * BK + BQ * 2 * BK);

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int tramo = blockIdx.x;
    const int sq_ = blockIdx.y / NH;                        // secuencia
    const int hh = blockIdx.y % NH;
    const int R = gridDim.y * BQ;
    const int bq = blockIdx.y * BQ;                         // primera fila de (secuencia, cabeza)
    const int N = nseq[sq_];
    const int wq = warp * NQW;                             // primera query del warp (en el bloque)
    const int kini = tramo * CHK;
    const int kfin = (kini + CHK) < N ? (kini + CHK) : N;
    const size_t KOFF = (size_t)CHK * NH * QD;
    const size_t EOFF = 2 * KOFF;
    const int hlim = bq + BQ;
    // Pagina sin tokens de esta secuencia: m = MINIT, S = 0 (la union la ignora).
    if (kini >= N) {
        if (lane == 0 && warp == 0)
            for (int r = 0; r < BQ; ++r) { out_m[(size_t)tramo * R + bq + r] = MINIT; out_s[(size_t)tramo * R + bq + r] = 0; }
        return;
    }
    // Pagina fisica: UNA carga de la tabla de bloques.
    const signed char* base = pool + (size_t)bt[(size_t)sq_ * BTS + tramo] * (size_t)BLK;

    const int gid = lane >> 2;
    const int tig = lane & 3;
    const int la_row = (lane & 7) + 8 * ((lane >> 3) & 1);
    const int la_col = 16 * (lane >> 4);
    const int lb_row = lane & 7;
    const int lb_mat = (lane >> 4) & 1;
    const int lb_col = 16 * ((lane >> 3) & 1);
    const int nthreads = NWARPS * 32;

    // Las 2 queries del hilo: columnas tig*2 + {0,1} del warp.
    int qr[2], lm[2], mq[2], dc[2];
#pragma unroll
    for (int e = 0; e < 2; ++e) {
        const int r = bq + wq + tig * 2 + e;
        const int vale = r < hlim;
        qr[e] = r;
        lm[e] = vale ? lim[r] : -1;
        mq[e] = vale ? mqb[r] : 1;
        dc[e] = vale ? dcap[r] : 0;
    }

    // Prologo: Q^T a shared (filas = queries del bloque, columnas = dims).
    for (int v = tid; v < BQ * CHQ; v += nthreads) {
        const int r = v / CHQ, c = (v % CHQ) * 16;
        const int gq = bq + r;
        const int ok = gq < hlim;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :: "r"(sdir(&sQ[r][swz(r, c, CHQ)])), "l"(Q + (size_t)(ok ? gq : bq) * QD + c), "r"(ok ? 16 : 0));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    // Q^T queda fijo: fragmentos B de Q.K por bloque de 32 dims (se cargan una vez).
    unsigned qb[8][2];
#pragma unroll
    for (int d = 0; d < 8; ++d) {
        const int rb = wq + lb_row;
        unsigned pb = sdir(&sQ[rb][swz(rb, d * 32 + lb_col, CHQ)]);
        unsigned basura[2];
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(qb[d][0]), "=r"(qb[d][1]), "=r"(basura[0]), "=r"(basura[1]) : "r"(pb));
    }

    int m[2] = {MINIT, MINIT};
    int S[2] = {0, 0};
    int oh[MT][4], ol[MT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int e = 0; e < 4; ++e) { oh[i][e] = 0; ol[i][e] = 0; }

#define SK18H_CARGA(ET, K0)                                                    \
    do {                                                                       \
        for (int v = tid; v < BK * CHQ; v += nthreads) {                       \
            const int r = v / CHQ, c = (v % CHQ) * 16;                         \
            const int key = (K0) + r;                                          \
            const int ok = key < kfin;                                         \
            if (DIAG != 5)                                                     \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"     \
                         :: "r"(sdir(&sK[ET][r][swz(r, c, CHQ)])),             \
                            "l"(base + ((size_t)((ok ? key : (K0)) - kini) * NH + hh) * QD + c), "r"(ok ? 16 : 0)); \
        }                                                                      \
        for (int v = tid; v < QD * CHV; v += nthreads) {                       \
            const int r = v / CHV, c = (v % CHV) * 16;                         \
            const int nk = kfin - ((K0) + c);                                  \
            const int sz = nk >= 16 ? 16 : (nk > 0 ? nk : 0);                  \
            if (DIAG != 4)                                                     \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"     \
                         :: "r"(sdir(&sV[ET][r][swz(r, c, CHV)])),             \
                            "l"(base + KOFF + ((size_t)hh * QD + r) * CHK + ((K0) - kini) + c), "r"(sz)); \
        }                                                                      \
        for (int v = tid; v < BK * NH * 4 / 16; v += nthreads) {               \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"     \
                         :: "r"(sdir(&sE[ET][v * 16])),                        \
                            "l"(base + EOFF + (size_t)((K0) - kini) * NH * 4 + v * 16), "r"(16)); \
        }                                                                      \
        __pipeline_commit();                                                   \
    } while (0)

#if SINCRONO
    // Carga sincronica por unidad: commit de la unidad actual y espera de ESE grupo antes de
    // leerla. El pipeline de 2 etapas (wait_prior(1)) y la variante wait_prior(0) con commit
    // adelantado daban oh/ol distintos entre corridas identicas (39 de 69 paginas a 57k).
    int buf = 0;
    for (int k0 = kini; k0 < kfin; k0 += BK) {
        SK18H_CARGA(0, k0);
        __pipeline_wait_prior(0);
        __syncthreads();
#else
    if (kini < kfin) { SK18H_CARGA(0, kini); } else { __pipeline_commit(); }
    int buf = 0;
    for (int k0 = kini; k0 < kfin; k0 += BK) {
        const int knext = k0 + BK;
        const int slot = (buf + 1) % 2;
        if (knext < kfin) { SK18H_CARGA(slot, knext); } else { __pipeline_commit(); }
        __pipeline_wait_prior(ESPERA);
        __syncthreads();
#endif

        // 1. Q.K: 4 fragmentos de 16 keys x 8 bloques de dims. C: filas = keys gid y
        //    gid+8 del fragmento, columnas = queries tig*2 + {0,1}.
        int zq[8][2];     // [key del hilo][query]
#pragma unroll
        for (int f = 0; f < 4; ++f) {
            int acc[4] = {0, 0, 0, 0};
#pragma unroll
            for (int d = 0; d < 8; ++d) {
                unsigned a[4];
                const int ra = f * 16 + la_row;
                unsigned pa = sdir(&sK[buf][ra][swz(ra, d * 32 + la_col, CHQ)]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(pa));
                if (DIAG != 1)
                asm volatile("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(qb[d][0]), "r"(qb[d][1]));
            }
#pragma unroll
            for (int hr = 0; hr < 2; ++hr)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int key = k0 + f * 16 + gid + hr * 8;
                    const int vis = key < kfin && key <= lm[e];
                    const int sk = vis ? (int)((short*)&sE[buf][((key - k0) * NH + hh) * 4])[0] : 0;
                    const int zz = (int)(((long long)acc[hr * 2 + e] * sk) >> ZSH);
                    zq[f * 2 + hr][e] = vis ? zz : PADZ;
                }
        }
        // 2. maximo corriente por query: local y mariposa sobre los 8 hilos (gid) de la query.
        int mn[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            int ml = m[e];
#pragma unroll
            for (int j = 0; j < 8; ++j) ml = ml > zq[j][e] ? ml : zq[j][e];
            int t = bfly(ml, 4);  ml = ml > t ? ml : t;
            t = bfly(ml, 8);      ml = ml > t ? ml : t;
            t = bfly(ml, 16);     ml = ml > t ? ml : t;
            mn[e] = ml;
        }
        // 3. reescalado en registro (S y acumuladores de las MISMAS queries del hilo).
        int sc[2];
        int sube_ = 0;
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            int dm = mn[e] - m[e];
            dm = dm < dc[e] ? dm : dc[e];
            sc[e] = pot2neg_q15((dm * mq[e] + 32768) >> 16);
            S[e] = ((S[e] >> 15) * sc[e]) + (((S[e] & 0x7fff) * sc[e]) >> 15);
            m[e] = mn[e];
            sube_ |= sc[e] != 32768;
        }
        // Reescalado de los acumuladores SOLO si en este warp algun maximo subio (raro despues
        // de las primeras unidades): el perfil por seccion lo puso en 23-33% del kernel.
        const int reesc_ = SALTEAR ? any_lanes(sube_) : 1;
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int c = 0; c < 4; ++c) if (DIAG != 2 && reesc_) {
                const int s_ = sc[c & 1];
                oh[i][c] = ((oh[i][c] >> 15) * s_) + (((oh[i][c] & 0x7fff) * s_) >> 15);
                ol[i][c] = ((ol[i][c] >> 15) * s_) + (((ol[i][c] & 0x7fff) * s_) >> 15);
            }
        // 4. pesos -> W (fila = query del bloque, columna = key de la unidad)
        int hi_or = 0, lo_or = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int col = (j >> 1) * 16 + gid + (j & 1) * 8;
                const int key = k0 + col;
                const int pad = zq[j][e] == PADZ;
                int d = m[e] - zq[j][e];
                d = d < dc[e] ? d : dc[e];
                const int w0 = DIAG == 8 ? 0 : pot2neg_q15((d * mq[e] + 32768) >> 16);
                const int w = pad ? 0 : ((w0 * 32639) >> 15);
                const int sv = pad ? 0 : (int)((short*)&sE[buf][((key - k0) * NH + hh) * 4])[1];
                const int wp = (w * sv + (1 << (VSH - 1))) >> VSH;
                const int hi = (wp + 128) >> 8;
                const int row = wq + tig * 2 + e;
                S[e] += w;
                hi_or |= hi != 0;
                lo_or |= (wp - (hi << 8)) != 0;
                if (DIAG != 6) {
                sW[row][swzB(row, 0, col & ~15, CHV) + (col & 15)] = (signed char)hi;
                sW[row][swzB(row, 1, col & ~15, CHV) + (col & 15)] = (signed char)(wp - (hi << 8));
                }
            }
        if (DIAG != 7) __syncthreads();
        // Planos de peso enteramente nulos en el warp -> su mma no suma nada: se saltea (exacto).
        const int usa_hi = SALTEAR ? any_lanes(hi_or) : 1;
        const int usa_lo = SALTEAR ? any_lanes(lo_or) : 1;

        // 5. w.v: A = V^T (16 fragmentos de dims x keys), B = W^T (queries del warp x keys).
#pragma unroll
        for (int kk = 0; kk < BK; kk += 32) {
            unsigned wh[2], wl[2];
            {
                const int rb = wq + lb_row;
                unsigned pb = sdir(&sW[rb][swzB(rb, lb_mat, kk + lb_col, CHV)]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(wh[0]), "=r"(wh[1]), "=r"(wl[0]), "=r"(wl[1]) : "r"(pb));
            }
#pragma unroll
            for (int i = 0; i < MT; ++i) if (usa_hi || usa_lo) {
                unsigned av[4];
                const int ra = i * 16 + la_row;
                unsigned pa = sdir(&sV[buf][ra][swz(ra, kk + la_col, CHV)]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(av[0]), "=r"(av[1]), "=r"(av[2]), "=r"(av[3]) : "r"(pa));
                if (DIAG != 3 && usa_hi)
                asm volatile("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+r"(oh[i][0]), "+r"(oh[i][1]), "+r"(oh[i][2]), "+r"(oh[i][3])
                             : "r"(av[0]), "r"(av[1]), "r"(av[2]), "r"(av[3]), "r"(wh[0]), "r"(wh[1]));
                if (DIAG != 3 && usa_lo)
                asm volatile("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+r"(ol[i][0]), "+r"(ol[i][1]), "+r"(ol[i][2]), "+r"(ol[i][3])
                             : "r"(av[0]), "r"(av[1]), "r"(av[2]), "r"(av[3]), "r"(wl[0]), "r"(wl[1]));
            }
        }
#if !SINCRONO
        buf = (buf + 1) % 2;
#endif
    }
#undef SK18H_CARGA

    // Epilogo: S total por query = suma de sus 8 hilos (mariposa sobre gid).
#pragma unroll
    for (int e = 0; e < 2; ++e) {
        int t = bfly(S[e], 4);  S[e] += t;
        t = bfly(S[e], 8);      S[e] += t;
        t = bfly(S[e], 16);     S[e] += t;
    }
#pragma unroll
    for (int e = 0; e < 2; ++e) {
        const int r = qr[e];
        if (gid == 0 && r < hlim) {
            out_m[(size_t)tramo * R + r] = m[e];
            out_s[(size_t)tramo * R + r] = S[e];
        }
    }
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int hr = 0; hr < 2; ++hr)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int r = qr[e];
                if (r < hlim) {
                    size_t o = ((size_t)tramo * R + r) * QD + i * 16 + gid + hr * 8;
                    out_hi[o] = oh[i][hr * 2 + e];
                    out_lo[o] = ol[i][hr * 2 + e];
                }
            }
}
