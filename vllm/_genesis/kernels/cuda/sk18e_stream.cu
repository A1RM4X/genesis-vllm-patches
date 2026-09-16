// SPDX-License-Identifier: Apache-2.0
//
// SK-18e — decode de atencion ENTERO en UNA pasada y UN lanzamiento, softmax en
// streaming a la FlashInfer (running max + reescalado), sin tabla y sin buffer M x K.
// Todo entero: no hay un solo flotante en el kernel.
//
// Grilla: x = tramo de keys (CHK keys, multiplo de 64), y = cabeza KV x bloque de
// 16 queries. 4 warps: cada warp es un tile de 64 dimensiones de la salida; el
// warp 0 ademas calcula Q.K y los pesos (los otros lo esperan y leen los pesos de
// shared: Q.K no se repite por warp).
//
// Por unidad de 64 keys (en el tramo):
//   1. warp 0: 8 subpasos de 8 keys x 8 mma s8 (32 dims c/u) -> 2 scores por hilo
//      (query g y g+8 son las filas del fragmento C; keys tig*2 + {0,1}).
//      z' = ((z >> 8) * skf[key]) >> 11, PAD si la key no es visible.
//   2. maximo corriente por query: max local y mariposa entre los 4 hilos de la
//      query (xor 1, xor 2), igual que FlashInfer.
//   3. factor de reescalado sc = 2^-(m_nuevo - m_viejo) en Q15 (misma exp de abajo);
//      el hilo tig==0 lo publica en shared; TODOS los warps lo aplican a sus
//      acumuladores int32: acc = (acc >> 15) * sc + (((acc & 0x7fff) * sc) >> 15).
//   4. peso por key: d = m - z', t = (d * mqb + 2^15) >> 16 (medios-log Q8),
//      n = t >> 8, f = t & 255, g = 2^-f/256 cuadratica en Q15, w0 = g*32639 >> 15,
//      w = redondeo(w0 / 2^n) (0 si n >= 16); w' = (w * svf + 2^14) >> 15 partido en
//      hi/lo; S += w; los bytes van a shared (layout B de SK-12: gate = hi, up = lo).
//   5. todos los warps: w.v = V^T (A, filas = dims, columnas = keys) x W^T (B, filas =
//      queries), 2 pasos k32, hi y lo -> acumuladores por (dim, query).
// Epilogo: out_hi/out_lo [NCH, NH*16*... ] por (tramo, fila de query, dim), y el warp
// 0 escribe m y S por (tramo, fila de query). La union de tramos va afuera (int64).
#include <cuda_fp16.h>
#include <cuda_pipeline.h>

#define BM 256          // dims por bloque (filas de A = V^T)
#define WM 64           // dims por warp
#define BN 16           // queries por bloque
#define WN 16
#define NWARPS 4
#define BK 64           // keys por unidad de streaming
#define MTILES (WM / 16)
#define NTILES (WN / 8)
#define QD 256          // dims de q/k
#define CHQ (QD / 16)   // chunks de 16 B por fila de Q/K
#define CHV (BK / 16)   // chunks por fila de V^T y de W
#define PADZ (-(1 << 28))
// DIAG: 0 normal; 1 sin mma de Q.K; 2 sin mma de w.v; 3 sin ninguno (solo carga/escalares/sync)
// V en paginas de BK keys [NH, K/BK, 256, BK] contiguas; keys una sola copia (los hilos
// "up" del ldmatrix x4 leen el mismo chunk).
#ifndef DIAG
#define DIAG 0
#endif
#define MINIT (-(1 << 30))

static_assert((BM / WM) * (BN / WN) == NWARPS, "grilla de warps");

// Q2^-x en Q15 con x = f/256: 1 - A x + B x^2 (extremos exactos, error max 0,27%).
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
// 2^-(t/256) en Q15 con 32768 = 1 (t en medios-log Q8), 0 si t >= 16*256.
__device__ __forceinline__ int pot2neg_q15(int t) {
    const int n = t >> 8, f = t & 255;
    const int g = 32768 - ((f * QA + 128) >> 8) + ((f * f * QB + 32768) >> 16);
    const int nn = n < 15 ? n : 15;
    const int r = (2 * g + (1 << nn)) >> (nn + 1);
    return n >= 16 ? 0 : r;
}

extern "C" __global__ void __launch_bounds__(NWARPS * 32)
sk18e_stream(
    const signed char* __restrict__ Q,      // [NH*MB, 256] queries int8 (MB multiplo de 16)
    const signed char* __restrict__ Kc,     // [NH*N, 256] keys int8
    const signed char* __restrict__ Vt,     // [NH, K/64, 256, 64] V int8 en paginas
    const short* __restrict__ skf,          // [NH*N]
    const short* __restrict__ svf,          // [NH*N]
    const int* __restrict__ lim,            // [NH*MB] ultima key visible (-1 = fila vacia)
    const int* __restrict__ mqb,            // [NH*MB] d -> medios-log Q8 (x 2^16)
    const int* __restrict__ dcap,           // [NH*MB] d >= dcap -> peso 0
    int* __restrict__ out_hi,               // [NCH, NH*MB, 256]
    int* __restrict__ out_lo,
    int* __restrict__ out_m,                // [NCH, NH*MB]
    int* __restrict__ out_s,
    int MB, int N, int K, int CHK, int NCH, int NH)
{
    extern __shared__ signed char _smem[];
    // Pipeline de 2 etapas (como SK-12): la unidad siguiente se copia mientras se calcula
    // la actual. 4 + 2*16 + 2*16 + 2 KB + 64 B = 71.744 B de shared dinamica.
    signed char (*sQ)[QD]         = (signed char (*)[QD]) _smem;                                  // 16 filas
    signed char (*sK)[BK][QD]     = (signed char (*)[BK][QD]) (_smem + 16 * QD);                  // [etapa][64 keys]
    signed char (*sV)[BM][BK]     = (signed char (*)[BM][BK]) (_smem + 16 * QD + 2 * BK * QD);    // [etapa][256 dims]
    signed char (*sW)[2 * BK]     = (signed char (*)[2 * BK]) (_smem + 16 * QD + 2 * BK * QD + 2 * BM * BK);
    int* sS = (int*) (_smem + 16 * QD + 2 * BK * QD + 2 * BM * BK + 16 * 2 * BK);              // sc por query

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm = warp * WM;                // dims de este warp
    const int tramo = blockIdx.x;
    const int TB = gridDim.y / NH;
    const int hh = blockIdx.y / TB;
    const int bq = (hh * TB + blockIdx.y % TB) * BN;      // fila global de query del bloque
    const int kini = tramo * CHK;
    const int kfin = (kini + CHK) < N ? (kini + CHK) : N;
    const size_t kh = (size_t)hh * N;                     // base de keys de la cabeza

    const int gid = lane >> 2;               // query g (y g+8) de este hilo
    const int tig = lane & 3;
    const int qrow[2] = {bq + gid, bq + gid + 8};

    const int la_row = (lane & 7) + 8 * ((lane >> 3) & 1);
    const int la_col = 16 * (lane >> 4);
    const int lb_row = lane & 7;
    const int lb_mat = (lane >> 4) & 1;
    const int lb_col = 16 * ((lane >> 3) & 1);
    const int nthreads = NWARPS * 32;

    // Parametros por query del hilo (warp 0 los usa).
    int lm[2], mq[2], dc[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int r = qrow[h];
        const int vale = r < (hh + 1) * MB ? 1 : 0;
        lm[h] = vale ? lim[r] : -1;
        mq[h] = vale ? mqb[r] : 1;
        dc[h] = vale ? dcap[r] : 0;
    }

    // Prologo: Q del bloque a shared (una vez; A de Q.K: filas = queries, 256 dims).
    for (int v = tid; v < 16 * CHQ; v += nthreads) {
        const int r = v / CHQ, c = (v % CHQ) * 16;
        const int gq = bq + r;
        const int ok = gq < (hh + 1) * MB;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :: "r"(sdir(&sQ[r][swz(r, c, CHQ)])), "l"(Q + (size_t)(ok ? gq : bq) * QD + c), "r"(ok ? 16 : 0));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    int m[2] = {MINIT, MINIT};
    int S[2] = {0, 0};
    int oh[MTILES][NTILES][4], ol[MTILES][NTILES][4];
#pragma unroll
    for (int i = 0; i < MTILES; ++i)
#pragma unroll
        for (int j = 0; j < NTILES; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) { oh[i][j][e] = 0; ol[i][j][e] = 0; }

#define SK18E_CARGA(ET, K0)                                                    \
    do {                                                                       \
        for (int v = tid; v < BK * CHQ; v += nthreads) {                       \
            const int r = v / CHQ, c = (v % CHQ) * 16;                         \
            const int key = (K0) + r;                                          \
            const int ok = key < kfin;                                         \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"     \
                         :: "r"(sdir(&sK[ET][r][swz(r, c, CHQ)])),             \
                            "l"(Kc + (kh + (ok ? key : (K0))) * QD + c), "r"(ok ? 16 : 0)); \
        }                                                                      \
        for (int v = tid; v < BM * CHV; v += nthreads) {                       \
            const int r = v / CHV, c = (v % CHV) * 16;                         \
            const int nk = kfin - ((K0) + c);                                  \
            const int sz = nk >= 16 ? 16 : (nk > 0 ? nk : 0);                  \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"     \
                         :: "r"(sdir(&sV[ET][r][swz(r, c, CHV)])),             \
                            "l"(Vt + ((((size_t)hh * (K / BK) + (K0) / BK) * QD + r) * BK + c)), "r"(sz)); \
        }                                                                      \
        __pipeline_commit();                                                   \
    } while (0)

    if (kini < kfin) { SK18E_CARGA(0, kini); } else { __pipeline_commit(); }
    int buf = 0;
    for (int k0 = kini; k0 < kfin; k0 += BK) {
        const int knext = k0 + BK;
        const int slot = (buf + 1) % 2;
        if (knext < kfin) { SK18E_CARGA(slot, knext); } else { __pipeline_commit(); }
        __pipeline_wait_prior(1);
        __syncthreads();

        // ── 1-4. warp 0: Q.K, maximo, reescalado y pesos ──────────────────────
        int zq[BK / 4][2];   // z' de las 16 keys del hilo por query (2 por subpaso x 8)
        if (warp == 0) {
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                int acc[4] = {0, 0, 0, 0};
#pragma unroll
                for (int d0 = 0; d0 < QD; d0 += 32) {
                    unsigned a[4], bg[2], bu[2];
                    {
                        unsigned pa = sdir(&sQ[la_row][swz(la_row, d0 + la_col, CHQ)]);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                     : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(pa));
                    }
                    {
                        const int rb = s * 8 + lb_row;
                        unsigned pb = sdir(&sK[buf][rb][swzB(rb, 0, d0 + lb_col, CHQ)]);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                     : "=r"(bg[0]), "=r"(bg[1]), "=r"(bu[0]), "=r"(bu[1]) : "r"(pb));
                    }
                    if (DIAG == 0 || DIAG == 2)
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
                                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                                 : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3])
                                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(bg[0]), "r"(bg[1]));
                }
                // c0,c1 -> query g, keys tig*2+{0,1}; c2,c3 -> query g+8
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const int key = k0 + s * 8 + tig * 2 + e;
                        const int vis = key < kfin && key <= lm[h];
                        const int sk = vis ? (int)skf[kh + key] : 0;
                        const int zz = ((acc[h * 2 + e] >> 8) * sk) >> 11;
                        zq[s * 2 + e][h] = vis ? zz : PADZ;
                    }
            }
            // 2. maximo corriente por query (mariposa entre los 4 hilos de la query)
            int mn[2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                int ml = m[h];
#pragma unroll
                for (int e = 0; e < 16; ++e) ml = ml > zq[e][h] ? ml : zq[e][h];
                mn[h] = ml;
            }
            {
                int t1[2];
                asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=r"(t1[0]) : "r"(mn[0]), "r"(1));
                asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=r"(t1[1]) : "r"(mn[1]), "r"(1));
                mn[0] = mn[0] > t1[0] ? mn[0] : t1[0]; mn[1] = mn[1] > t1[1] ? mn[1] : t1[1];
                asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=r"(t1[0]) : "r"(mn[0]), "r"(2));
                asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=r"(t1[1]) : "r"(mn[1]), "r"(2));
                mn[0] = mn[0] > t1[0] ? mn[0] : t1[0]; mn[1] = mn[1] > t1[1] ? mn[1] : t1[1];
            }
            // 3. factor de reescalado (Q15) y S
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                int dm = mn[h] - m[h];
                dm = dm < dc[h] ? dm : dc[h];
                const int sc = pot2neg_q15((dm * mq[h] + 32768) >> 16);
                if (tig == 0) sS[gid + h * 8] = sc;
                S[h] = ((S[h] >> 15) * sc) + (((S[h] & 0x7fff) * sc) >> 15);
                m[h] = mn[h];
            }
            // 4. pesos -> shared (fila de W = query, columna = key de la unidad)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int e = 0; e < 16; ++e) {
                    const int key = k0 + (e / 2) * 8 + tig * 2 + (e & 1);
                    int d = m[h] - zq[e][h];
                    d = d < dc[h] ? d : dc[h];
                    const int t = (d * mq[h] + 32768) >> 16;
                    const int w0 = pot2neg_q15(t);
                    const int w = zq[e][h] == PADZ ? 0 : ((w0 * 32639) >> 15);
                    const int vis = zq[e][h] != PADZ;
                    const int sv = vis ? (int)svf[kh + key] : 0;
                    const int wp = (w * sv + 16384) >> 15;
                    const int hi = (wp + 128) >> 8;
                    const int col = (e / 2) * 8 + tig * 2 + (e & 1);
                    const int row = gid + h * 8;
                    S[h] += w;
                    sW[row][swzB(row, 0, col & ~15, CHV) + (col & 15)] = (signed char)hi;
                    sW[row][swzB(row, 1, col & ~15, CHV) + (col & 15)] = (signed char)(wp - (hi << 8));
                }
        }
        __syncthreads();

        // ── 3'. reescalado de los acumuladores (todos los warps) ──────────────
#pragma unroll
        for (int j = 0; j < NTILES; ++j) {
            const int sca = sS[j * 8 + tig * 2], scb = sS[j * 8 + tig * 2 + 1];
#pragma unroll
            for (int i = 0; i < MTILES; ++i)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int sc = (e & 1) ? scb : sca;
                    oh[i][j][e] = ((oh[i][j][e] >> 15) * sc) + (((oh[i][j][e] & 0x7fff) * sc) >> 15);
                    ol[i][j][e] = ((ol[i][j][e] >> 15) * sc) + (((ol[i][j][e] & 0x7fff) * sc) >> 15);
                }
        }

        // ── 5. w.v: A = V^T (filas dims del warp, columnas keys), B = W^T ─────
#pragma unroll
        for (int kk = 0; kk < BK; kk += 32) {
            unsigned av[MTILES][4], wh[NTILES][2], wl[NTILES][2];
#pragma unroll
            for (int i = 0; i < MTILES; ++i) {
                const int ra = wm + i * 16 + la_row;
                unsigned pa = sdir(&sV[buf][ra][swz(ra, kk + la_col, CHV)]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(av[i][0]), "=r"(av[i][1]), "=r"(av[i][2]), "=r"(av[i][3]) : "r"(pa));
            }
#pragma unroll
            for (int j = 0; j < NTILES; ++j) {
                const int rb = j * 8 + lb_row;
                unsigned pb = sdir(&sW[rb][swzB(rb, lb_mat, kk + lb_col, CHV)]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(wh[j][0]), "=r"(wh[j][1]), "=r"(wl[j][0]), "=r"(wl[j][1]) : "r"(pb));
            }
#pragma unroll
            for (int i = 0; i < MTILES; ++i)
#pragma unroll
                for (int j = 0; j < NTILES; ++j) if (DIAG == 0 || DIAG == 1) {
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
                                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                                 : "+r"(oh[i][j][0]), "+r"(oh[i][j][1]), "+r"(oh[i][j][2]), "+r"(oh[i][j][3])
                                 : "r"(av[i][0]), "r"(av[i][1]), "r"(av[i][2]), "r"(av[i][3]), "r"(wh[j][0]), "r"(wh[j][1]));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
                                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                                 : "+r"(ol[i][j][0]), "+r"(ol[i][j][1]), "+r"(ol[i][j][2]), "+r"(ol[i][j][3])
                                 : "r"(av[i][0]), "r"(av[i][1]), "r"(av[i][2]), "r"(av[i][3]), "r"(wl[j][0]), "r"(wl[j][1]));
                }
        }
        __syncthreads();
        buf = (buf + 1) % 2;
    }
#undef SK18E_CARGA

    // Epilogo. S total por query = suma de los 4 hilos de la query (mariposa).
    if (warp == 0) {
        int s1[2];
        asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=r"(s1[0]) : "r"(S[0]), "r"(1));
        asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=r"(s1[1]) : "r"(S[1]), "r"(1));
        S[0] += s1[0]; S[1] += s1[1];
        asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=r"(s1[0]) : "r"(S[0]), "r"(2));
        asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=r"(s1[1]) : "r"(S[1]), "r"(2));
        S[0] += s1[0]; S[1] += s1[1];
        if (tig == 0) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = qrow[h];
                if (r < (hh + 1) * MB) {
                    out_m[(size_t)tramo * NH * MB + r] = m[h];
                    out_s[(size_t)tramo * NH * MB + r] = S[h];
                }
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < MTILES; ++i)
#pragma unroll
        for (int hr = 0; hr < 2; ++hr) {
            const int dim = wm + i * 16 + gid + hr * 8;
#pragma unroll
            for (int j = 0; j < NTILES; ++j)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int r = bq + j * 8 + tig * 2 + e;
                    if (r < (hh + 1) * MB) {
                        size_t o = ((size_t)tramo * NH * MB + r) * QD + dim;
                        out_hi[o] = oh[i][j][hr * 2 + e];
                        out_lo[o] = ol[i][j][hr * 2 + e];
                    }
                }
        }
}
