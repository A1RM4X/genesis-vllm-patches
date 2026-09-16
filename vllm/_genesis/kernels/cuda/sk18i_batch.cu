// SPDX-License-Identifier: Apache-2.0
//
// SK-18i — decode entero con KV INT4 (mma m16n8k64 s4.s4), misma estructura de dos pasadas
// que sk18h_batch2 (maximo exacto por pagina y despues pesos contra ese maximo fijo: nunca
// se leen/escriben los acumuladores del mma con operaciones de lane).
//
// Todo el layout de bytes es el mismo que el camino int8: un byte = dos nibbles, asi que
// una unidad de BK = 128 keys ocupa los mismos bytes que 64 keys en int8 y los fragmentos
// del mma s4 k64 tienen el layout de bytes del s8 k32.
//
// Bloque fisico (BLK bytes):
//   [0, KOFF)      K nibbles [BS][NH][128]     (dim par en el nibble bajo)
//   [KOFF, EOFF)   V nibbles [NH][256][BS/2]   (token par en el nibble bajo)
//   [EOFF, ...)    escalas   [BS][NH][8]: kmax int16 | rk0..rk3 uint8 | svf int16
// Logit (por grupo g de 64 dims, acumulador propio del mma):
//   z = (sum_g acc_g * rq_g * rk_g * kmax) >> ZSH4
// Pesos en 3 planos de nibbles con signo (12 bits, wp <= 1911):
//   O = sum_p (w_p . V) << (4 p)
#include <cuda_fp16.h>
#include <cuda_pipeline.h>

#define QD 256          // dims de q/k/v
#define QB_ (QD / 2)    // bytes por fila de q/k
#define NQW 8
#ifndef NWARPS
#define NWARPS 4
#endif
#define BQ (NQW * NWARPS)
#define BK 128          // keys por unidad (64 bytes de V, 128 filas de K)
#define BKB (BK / 2)    // bytes de V por unidad
#define MT (QD / 16)
#define CHQ (QB_ / 16)  // 8 trozos de 16 bytes por fila de q/k
#define CHV (BKB / 16)  // 4
#define PADZ (-(1 << 28))
#define NHMAX 2
#define MINIT (-(1 << 30))
#define WCAP 1911       // 7 * (1 + 16 + 256): tope de wp con 3 planos de nibbles
// DIAG (solo diagnostico): 1 = fragmentos de V constantes (aisla sV), 2 = pesos constantes
// (aisla sWa/sWb), 3 = las dos cosas.
#ifndef DIAG
#define DIAG 0
#endif
// QPLANOS=2: q int8 partido en dos planos de nibbles (q = 16*alto + bajo); duplica el mma de
// Q.K y recupera la precision de q (con 1 plano q va en nibbles como la K).
#ifndef QPLANOS
#define QPLANOS 1
#endif
// La pasada del maximo puede usar solo el plano alto (PLANOS_A=1): el maximo sale un poco
// chico y la pasada B recorta d a 0 (peso 1) en esas keys. Con QPLANOS=1 son la misma cosa.
// ESPERA: grupos de cp.async pendientes al entrar a la unidad (0 = esperar todo)
#ifndef ESPERA
#define ESPERA 1
#endif
#ifndef PLANOS_A
#define PLANOS_A QPLANOS
#endif

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

extern "C" __global__ void __launch_bounds__(NWARPS * 32)
sk18i_batch(
    const unsigned char* __restrict__ Q,    // [B, NH, 32, 128] nibbles
    const int* __restrict__ rq,             // [B, NH, 32, 4] ratios de grupo de q
    const int* __restrict__ sq,             // [B, NH, 32, QPLANOS, 4] rq_g * suma de nibbles de q
    const unsigned char* __restrict__ pool,
    const int* __restrict__ bt,             // [B, BTS]
    const int* __restrict__ nseq,           // [B]
    const int* __restrict__ lim,            // [B, NH, 32]
    const int* __restrict__ mqb,
    const int* __restrict__ dcap,
    int* __restrict__ out_hi,               // [NCH, R, 256]
    int* __restrict__ out_lo,
    int* __restrict__ out_m,                // [NCH, R]
    int* __restrict__ out_s,
    int* __restrict__ out_c,                // [NCH, R] correccion del cero de V: sum_k wp * vzp
    int BLK, int BTS, int CHK, int NCH, int NH, int ZSH4, int VSH,
    const int* __restrict__ dueno,          // [B * PAGS] ranuras del espejo int8 (VENT > 0)
    int PAGS)
{
    extern __shared__ signed char _smem[];
    const int SQB = BQ * QPLANOS * QB_;
    signed char (*sQ)[QB_]        = (signed char (*)[QB_]) _smem;
    signed char (*sK)[BK][QB_]    = (signed char (*)[BK][QB_]) (_smem + SQB);
    signed char (*sV)[QD][BKB]    = (signed char (*)[QD][BKB]) (_smem + SQB + 2 * BK * QB_);
    signed char (*sWa)[2 * BKB]   = (signed char (*)[2 * BKB]) (_smem + SQB + 2 * BK * QB_ + 2 * QD * BKB);
    signed char (*sWb)[2 * BKB]   = (signed char (*)[2 * BKB]) (_smem + SQB + 2 * BK * QB_ + 2 * QD * BKB + BQ * 2 * BKB);
    unsigned char (*sE)[BK * NHMAX * 8] = (unsigned char (*)[BK * NHMAX * 8])
        (_smem + SQB + 2 * BK * QB_ + 2 * QD * BKB + 2 * BQ * 2 * BKB);

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int tramo = blockIdx.x;
    const int sq_ = blockIdx.y / NH;
    const int hh = blockIdx.y % NH;
    const int R = gridDim.y * BQ;
    const int bq = blockIdx.y * BQ;
    const int N = nseq[sq_];
    const int wq = warp * NQW;
    const int kini = tramo * CHK;
    const int kfin = (kini + CHK) < N ? (kini + CHK) : N;
    const size_t KOFF = (size_t)CHK * NH * QB_;
    const size_t EOFF = 2 * KOFF;
    const int hlim = bq + BQ;
    if (kini >= N) {
        if (lane == 0 && warp == 0)
            for (int r = 0; r < BQ; ++r) {
                out_m[(size_t)tramo * R + bq + r] = MINIT;
                out_s[(size_t)tramo * R + bq + r] = 0;
                out_c[(size_t)tramo * R + bq + r] = 0;
            }
        return;
    }
    const int fis = bt[(size_t)sq_ * BTS + tramo];
    // Ventana reciente: si la pagina esta espejada en int8, la hace sk18h_batch2 (HIB)
    if (PAGS > 0 && dueno[sq_ * PAGS + (tramo % PAGS)] == fis) return;
    const unsigned char* base = pool + (size_t)fis * (size_t)BLK;

    const int gid = lane >> 2;
    const int tig = lane & 3;
    const int la_row = (lane & 7) + 8 * ((lane >> 3) & 1);
    const int la_col = 16 * (lane >> 4);
    const int lb_row = lane & 7;
    const int lb_mat = (lane >> 4) & 1;
    const int lb_col = 16 * ((lane >> 3) & 1);
    const int nthreads = NWARPS * 32;

    int qr_[2], lm[2], mq[2], dc[2], rqg[2][4], sqg[2][QPLANOS][4];
#pragma unroll
    for (int e = 0; e < 2; ++e) {
        const int r = bq + wq + tig * 2 + e;
        const int vale = r < hlim;
        qr_[e] = r;
        lm[e] = vale ? lim[r] : -1;
        mq[e] = vale ? mqb[r] : 1;
        dc[e] = vale ? dcap[r] : 0;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            rqg[e][g] = vale ? rq[(size_t)r * 4 + g] : 0;
#pragma unroll
            for (int pl = 0; pl < QPLANOS; ++pl)
                sqg[e][pl][g] = vale ? sq[((size_t)r * QPLANOS + pl) * 4 + g] : 0;
        }
    }

    // Q^T (nibbles) a shared
    for (int v = tid; v < BQ * QPLANOS * CHQ; v += nthreads) {
        const int r = v / CHQ, c = (v % CHQ) * 16;        // r = fila (query * QPLANOS + plano)
        const int gq = bq + r / QPLANOS;
        const int ok = gq < hlim;
        const size_t src = (size_t)((ok ? gq : bq) * QPLANOS + (r % QPLANOS)) * QB_;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :: "r"(sdir(&sQ[r][swz(r, c, CHQ)])), "l"(Q + src + c), "r"(ok ? 16 : 0));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    unsigned qb[QPLANOS][4][2];
#pragma unroll
    for (int pl = 0; pl < QPLANOS; ++pl)
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            const int rb = (wq + lb_row) * QPLANOS + pl;
            unsigned pb = sdir(&sQ[rb][swz(rb, d * 32 + lb_col, CHQ)]);
            unsigned basura[2];
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(qb[pl][d][0]), "=r"(qb[pl][d][1]), "=r"(basura[0]), "=r"(basura[1]) : "r"(pb));
        }

    int m[2] = {MINIT, MINIT};
    int S[2] = {0, 0};
    int C[2] = {0, 0};                       // sum_k wp * vzp (cero de V, igual para toda dim)
    int o0[MT][4], o1[MT][4], o2[MT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int e = 0; e < 4; ++e) { o0[i][e] = 0; o1[i][e] = 0; o2[i][e] = 0; }

#define SK18I_CARGA_K(ET, K0)                                                  \
    do {                                                                       \
        for (int v = tid; v < BK * CHQ; v += nthreads) {                       \
            const int r = v / CHQ, c = (v % CHQ) * 16;                         \
            const int key = (K0) + r;                                          \
            const int ok = key < kfin;                                         \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"     \
                         :: "r"(sdir(&sK[ET][r][swz(r, c, CHQ)])),             \
                            "l"(base + ((size_t)((ok ? key : (K0)) - kini) * NH + hh) * QB_ + c), "r"(ok ? 16 : 0)); \
        }                                                                      \
        for (int v = tid; v < BK * NH * 8 / 16; v += nthreads) {               \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"     \
                         :: "r"(sdir(&sE[ET][v * 16])),                        \
                            "l"(base + EOFF + (size_t)((K0) - kini) * NH * 8 + v * 16), "r"(16)); \
        }                                                                      \
    } while (0)

#define SK18I_CARGA_V(ET, K0)                                                  \
    do {                                                                       \
        for (int v = tid; v < QD * CHV; v += nthreads) {                       \
            const int r = v / CHV, c = (v % CHV) * 16;                         \
            const int nk = kfin - ((K0) + 2 * c);                              \
            const int nb_ = (nk + 1) >> 1;                                     \
            const int sz = nb_ >= 16 ? 16 : (nb_ > 0 ? nb_ : 0);               \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"     \
                         :: "r"(sdir(&sV[ET][r][swz(r, c, CHV)])),             \
                            "l"(base + KOFF + ((size_t)hh * QD + r) * (CHK / 2) + (((K0) - kini) >> 1) + c), "r"(sz)); \
        }                                                                      \
    } while (0)

// Q.K de la fila de keys f (16 keys) -> zq[2][2] (hr, e)
#define SK18I_QK_F(ET, K0, f, NP)                                                  \
        int zq[2][2];                                                          \
        {                                                                      \
            long long zs[2][2] = {{0, 0}, {0, 0}};                             \
            int acc[4][4];                                                     \
            _Pragma("unroll") for (int pl = 0; pl < (NP); ++pl) {              \
                _Pragma("unroll") for (int d = 0; d < 4; ++d) {                \
                    acc[d][0] = 0; acc[d][1] = 0; acc[d][2] = 0; acc[d][3] = 0;\
                    unsigned a[4];                                             \
                    const int ra = (f) * 16 + la_row;                          \
                    unsigned pa = sdir(&sK[ET][ra][swz(ra, d * 32 + la_col, CHQ)]); \
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" \
                                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(pa)); \
                    asm volatile("mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.s4.s32 " \
                                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n" \
                                 : "+r"(acc[d][0]), "+r"(acc[d][1]), "+r"(acc[d][2]), "+r"(acc[d][3]) \
                                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), \
                                    "r"(qb[(NP) == 1 ? QPLANOS - 1 : pl][d][0]), "r"(qb[(NP) == 1 ? QPLANOS - 1 : pl][d][1])); \
                }                                                              \
                _Pragma("unroll") for (int hr = 0; hr < 2; ++hr)               \
                    _Pragma("unroll") for (int e = 0; e < 2; ++e) {            \
                        const unsigned char* pe = &sE[ET][(((K0) + (f) * 16 + gid + hr * 8 - (K0)) * NH + hh) * 8]; \
                        int t = 0;                                             \
                        _Pragma("unroll") for (int d = 0; d < 4; ++d) {        \
                            const int b = (int)pe[2 + d];                      \
                            const int rk = (b & 15) + 1, zp = ((b >> 4) & 15) - 8; \
                            t += rk * (rqg[e][d] * acc[d][hr * 2 + e]          \
                                       - zp * sqg[e][(NP) == 1 ? QPLANOS - 1 : pl][d]); \
                        }                                                      \
                        zs[hr][e] += ((NP) == 2 && pl) ? ((long long)t << 4) : (long long)t; \
                    }                                                          \
            }                                                                  \
            _Pragma("unroll") for (int hr = 0; hr < 2; ++hr)                   \
                _Pragma("unroll") for (int e = 0; e < 2; ++e) {                \
                    const int key = (K0) + (f) * 16 + gid + hr * 8;            \
                    const int vis = key < kfin && key <= lm[e];                \
                    const unsigned char* pe = &sE[ET][((key - (K0)) * NH + hh) * 8]; \
                    const int kmax = (int)(pe[0] | (pe[1] << 8)) & 4095;       \
                    const int zz = (int)(((long long)zs[hr][e] * kmax) >> ZSH4); \
                    zq[hr][e] = vis ? zz : PADZ;                               \
                }                                                              \
        }

    // ── Pasada A: maximo exacto de la pagina (solo K + escalas + Q.K) ──
    int buf = 0;
    SK18I_CARGA_K(0, kini);
    __pipeline_commit();
    for (int k0 = kini; k0 < kfin; k0 += BK) {
        if (k0 + BK < kfin) { SK18I_CARGA_K((buf + 1) % 2, k0 + BK); }
        __pipeline_commit();
        __pipeline_wait_prior(ESPERA);
        __syncthreads();
#pragma unroll
        for (int f = 0; f < BK / 16; ++f) {
            SK18I_QK_F(buf, k0, f, PLANOS_A)
#pragma unroll
            for (int e = 0; e < 2; ++e)
#pragma unroll
                for (int hr = 0; hr < 2; ++hr) {
                    const int zv = (PLANOS_A == 1 && QPLANOS == 2) ? (zq[hr][e] << 4) : zq[hr][e];
                    m[e] = (zq[hr][e] != PADZ && zv > m[e]) ? zv : m[e];
                }
        }
        // Nadie puede emitir los cp.async de la unidad siguiente mientras otro warp siga
        // leyendo esta etapa: las copias se emiten ARRIBA del lazo, antes de la espera.
        __syncthreads();
        buf = (buf + 1) % 2;
    }
    __pipeline_wait_prior(0);
#pragma unroll
    for (int e = 0; e < 2; ++e) {
        int t = bfly(m[e], 4);  m[e] = m[e] > t ? m[e] : t;
        t = bfly(m[e], 8);      m[e] = m[e] > t ? m[e] : t;
        t = bfly(m[e], 16);     m[e] = m[e] > t ? m[e] : t;
    }

    // ── Pasada B: pesos contra el maximo fijo y w.v ──
    buf = 0;
    __syncthreads();
    SK18I_CARGA_K(0, kini);
    SK18I_CARGA_V(0, kini);
    __pipeline_commit();
    for (int k0 = kini; k0 < kfin; k0 += BK) {
        if (k0 + BK < kfin) { SK18I_CARGA_K((buf + 1) % 2, k0 + BK); SK18I_CARGA_V((buf + 1) % 2, k0 + BK); }
        __pipeline_commit();
        __pipeline_wait_prior(ESPERA);
        __syncthreads();
#pragma unroll
        for (int f = 0; f < BK / 16; ++f) {
            SK18I_QK_F(buf, k0, f, QPLANOS)
#pragma unroll
            for (int hr = 0; hr < 2; ++hr)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int col = f * 16 + gid + hr * 8;         // key dentro de la unidad
                    const int key = k0 + col;
                    const int pad = zq[hr][e] == PADZ;
                    int d = m[e] - zq[hr][e];
                    d = d > 0 ? d : 0;                             // el maximo puede ser aproximado
                    d = d < dc[e] ? d : dc[e];
                    const int w0 = pot2neg_q15((d * mq[e] + 32768) >> 16);
                    const int w = pad ? 0 : ((w0 * WCAP) >> 15);
                    const unsigned char* pe = &sE[buf][(col * NH + hh) * 8];
                    const int sv = pad ? 0 : (int)(pe[6] | (pe[7] << 8));
                    const int vzp = (int)((pe[1] >> 4) & 15) - 8;
                    int wp = (w * sv + (1 << (VSH - 1))) >> VSH;
                    wp = wp < WCAP ? wp : WCAP;
                    S[e] += w;
                    C[e] += wp * vzp;
                    const int n0 = ((wp + 8) & 15) - 8;
                    const int t1 = (wp - n0) >> 4;
                    const int n1 = ((t1 + 8) & 15) - 8;
                    const int n2 = (t1 - n1) >> 4;
                    const int emp = (n0 & 15) | ((n1 & 15) << 4) | ((n2 & 15) << 8);
                    const int otro = bfly(emp, 4);                 // nibbles de la key col^1
                    if ((gid & 1) == 0) {                          // el lane par escribe el byte
                        const int bcol = col >> 1;
                        const int row = wq + tig * 2 + e;
                        const int off0 = swzB(row, 0, bcol & ~15, CHV) + (bcol & 15);
                        const int off1 = swzB(row, 1, bcol & ~15, CHV) + (bcol & 15);
                        sWa[row][off0] = (signed char)(((emp & 15)) | ((otro & 15) << 4));
                        sWa[row][off1] = (signed char)(((emp >> 4) & 15) | (((otro >> 4) & 15) << 4));
                        sWb[row][off0] = (signed char)(((emp >> 8) & 15) | (((otro >> 8) & 15) << 4));
                        sWb[row][off1] = 0;
                    }
                }
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BKB; kk += 32) {
            unsigned wa[4], wb[4];
            {
                const int rb = wq + lb_row;
                unsigned pa = sdir(&sWa[rb][swzB(rb, lb_mat, kk + lb_col, CHV)]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(wa[0]), "=r"(wa[1]), "=r"(wa[2]), "=r"(wa[3]) : "r"(pa));
                unsigned pb = sdir(&sWb[rb][swzB(rb, lb_mat, kk + lb_col, CHV)]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(wb[0]), "=r"(wb[1]), "=r"(wb[2]), "=r"(wb[3]) : "r"(pb));
                if (DIAG == 2 || DIAG == 3) {
                    wa[0] = 0x11111111u; wa[1] = 0x11111111u; wa[2] = 0x11111111u; wa[3] = 0x11111111u;
                    wb[0] = 0x11111111u; wb[1] = 0x11111111u; wb[2] = 0x11111111u; wb[3] = 0x11111111u;
                }
            }
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                unsigned av[4];
                const int ra = i * 16 + la_row;
                unsigned pa = sdir(&sV[buf][ra][swz(ra, kk + la_col, CHV)]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(av[0]), "=r"(av[1]), "=r"(av[2]), "=r"(av[3]) : "r"(pa));
                if (DIAG == 1 || DIAG == 3) { av[0] = 0x11111111u; av[1] = 0x22222222u; av[2] = 0x33333333u; av[3] = 0x44444444u; }
                asm volatile("mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.s4.s32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+r"(o0[i][0]), "+r"(o0[i][1]), "+r"(o0[i][2]), "+r"(o0[i][3])
                             : "r"(av[0]), "r"(av[1]), "r"(av[2]), "r"(av[3]), "r"(wa[0]), "r"(wa[1]));
                asm volatile("mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.s4.s32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+r"(o1[i][0]), "+r"(o1[i][1]), "+r"(o1[i][2]), "+r"(o1[i][3])
                             : "r"(av[0]), "r"(av[1]), "r"(av[2]), "r"(av[3]), "r"(wa[2]), "r"(wa[3]));
                asm volatile("mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.s4.s32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+r"(o2[i][0]), "+r"(o2[i][1]), "+r"(o2[i][2]), "+r"(o2[i][3])
                             : "r"(av[0]), "r"(av[1]), "r"(av[2]), "r"(av[3]), "r"(wb[0]), "r"(wb[1]));
            }
        }
        __syncthreads();
        buf = (buf + 1) % 2;
    }

    // Epilogo
#pragma unroll
    for (int e = 0; e < 2; ++e) {
        int t = bfly(S[e], 4);  S[e] += t;
        t = bfly(S[e], 8);      S[e] += t;
        t = bfly(S[e], 16);     S[e] += t;
        t = bfly(C[e], 4);      C[e] += t;
        t = bfly(C[e], 8);      C[e] += t;
        t = bfly(C[e], 16);     C[e] += t;
    }
#pragma unroll
    for (int e = 0; e < 2; ++e) {
        const int r = qr_[e];
        if (gid == 0 && r < hlim) {
            out_m[(size_t)tramo * R + r] = m[e];
            out_s[(size_t)tramo * R + r] = S[e];
            out_c[(size_t)tramo * R + r] = C[e];
        }
    }
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int hr = 0; hr < 2; ++hr)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int r = qr_[e];
                if (r < hlim) {
                    size_t o = ((size_t)tramo * R + r) * QD + i * 16 + gid + hr * 8;
                    // mismo formato que el camino int8: O = hi * 256 + lo
                    const int O = o0[i][hr * 2 + e] + (o1[i][hr * 2 + e] << 4) + (o2[i][hr * 2 + e] << 8);
                    const int hi2 = O >> 8;
                    out_hi[o] = hi2;
                    out_lo[o] = O - (hi2 << 8);
                }
            }
}
