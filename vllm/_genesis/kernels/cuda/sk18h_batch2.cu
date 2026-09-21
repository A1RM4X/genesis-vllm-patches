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
#ifndef BK
#define BK 64           // keys por unidad (2 etapas de shared para el pipeline de carga)
#endif
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

// HIB=1: instancia ESPEJO del camino hibrido int4 (ver sk18i_lado.cu). Procesa solo las paginas
// que estan espejadas en el pool int8 de la ventana reciente, convierte su maximo a las unidades
// de z del camino int4 (factor mqb8/mqb4 por fila) y baja su O en ESC8 bits (la referencia 2^ev
// del espejo es 2^ESC8 mas chica). WCAPW iguala la escala de los pesos con la del int4.
// ARBOL=1: verificacion de un ARBOL de borrador. Las keys hasta el ancla (abase) se ven siempre;
// entre los tokens nuevos, la query ve solo a sus ancestros y a si misma: bit (key-abase-1) de
// amask. Con amask = (1<<j)-1 es exactamente la causalidad de siempre (prueba de regresion).
#ifndef ARBOL
#define ARBOL 0
#endif
#ifndef HIB
#define HIB 0
#endif
#ifndef ESC8
#define ESC8 3
#endif
// Trozos por pagina espejada: con 4 bloques (2 paginas x 2 cabezas) el kernel queda limitado por
// latencia (114 us por capa). Cada trozo toma CHK/SCH keys y escribe en una ranura de salida
// propia al final del espacio de paginas (las reserva el runtime).
#ifndef SCH
#define SCH 8
#endif
#ifndef WCAPW
#define WCAPW 32639
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

// OR/max de un entero chico sobre los 32 lanes del warp (5 etapas de mariposa)
__device__ __forceinline__ int any_lanes(int x) {
    for (int k = 1; k < 32; k <<= 1) { const int t = bfly(x, k); x = x > t ? x : t; }
    return x;
}

extern "C" __global__ void __launch_bounds__(NWARPS * 32)
sk18h_batch2(
    const signed char* __restrict__ Q,      // [B, NH, 32, 256]
    const signed char* __restrict__ pool,   // bloques fisicos contiguos de BLK bytes
    const int* __restrict__ bt,             // [B, BTS] tabla de bloques
    const int* __restrict__ nseq,           // [B] tokens en KV por secuencia
    const int* __restrict__ lim,            // [B, NH, 32] ultima key visible (-1 = fila vacia)
    const int* __restrict__ mqb,            // [B, NH, 32]
    const int* __restrict__ dcap,           // [B, NH, 32]
#if ARBOL
    const int* __restrict__ abase,          // [B, NH, 32] posicion del ancla (ultima key siempre visible)
    const int* __restrict__ amask,          // [B, NH, 32] bits de ancestros entre los tokens nuevos
#endif
    int* __restrict__ out_hi,               // [NCH, R, 256]  R = B*NH*32
    int* __restrict__ out_lo,
    int* __restrict__ out_m,                // [NCH, R]
    int* __restrict__ out_s,
    int BLK, int BTS, int CHK, int NCH, int NH, int SUB, int ZSH, int VSH   // CHK = BS; BLK = bytes por bloque
#if HIB
    , const int* __restrict__ dueno         // [B * PAGS] bloque fisico que ocupa cada ranura
    , const int* __restrict__ mqb4          // [R] mqb del camino int4 (para igualar unidades)
    , int* __restrict__ out_c               // [NCH, R] correccion del cero de V (0 en el espejo)
    , int PAGS
#endif
    )
{
    // Shared: Q^T [BQ][256] | K [2][64][256] | V^T [2][256][64] | W [BQ][128]
    extern __shared__ signed char _smem[];
    signed char (*sQ)[QD]         = (signed char (*)[QD]) _smem;
    signed char (*sK)[BK][QD]     = (signed char (*)[BK][QD]) (_smem + BQ * QD);
    // Q^T | K [2 etapas] | V^T [2 etapas] | W | escalas [2 etapas]
    signed char (*sV)[QD][BK]     = (signed char (*)[QD][BK]) (_smem + BQ * QD + 2 * BK * QD);
    signed char (*sW)[2 * BK]     = (signed char (*)[2 * BK]) (_smem + BQ * QD + 2 * BK * QD + 2 * QD * BK);
    // Escalas de la unidad (skf, svf) en shared: antes eran cargas globales por key
    // (long_scoreboard 0,93 contra 0,23 de FlashInfer en ncu).
    signed char (*sE)[BK * NHMAX * 4] = (signed char (*)[BK * NHMAX * 4]) (_smem + BQ * QD + 2 * BK * QD + 2 * QD * BK + BQ * 2 * BK);

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
#if HIB   // la grilla del espejo tiene PAGS paginas x SCH trozos: son las ULTIMAS de cada secuencia
    const int sq_ = blockIdx.y / (NH * SUB);
    const int trozo = blockIdx.x % SCH;
    const int tramo = ((nseq[sq_] - 1) / CHK) - (PAGS - 1) + (blockIdx.x / SCH);
    const int ranura_out = NCH + blockIdx.x;      // ranuras extra, despues de las paginas reales
    if (tramo < 0 || tramo >= NCH) return;
#else
    const int tramo = blockIdx.x;
    const int sq_ = blockIdx.y / (NH * SUB);
#endif                        // secuencia
    const int hh = (blockIdx.y / SUB) % NH;
    const int R = gridDim.y * BQ;
    const int bq = blockIdx.y * BQ;                         // primera fila de (secuencia, cabeza)
    const int N = nseq[sq_];
    const int wq = warp * NQW;                             // primera query del warp (en el bloque)
    const int kbase = tramo * CHK;          // inicio de la PAGINA (los offsets del bloque van con esto)
#if HIB
    // El trozo se redondea a multiplo de BK (y por lo tanto de 16): los cp.async de V leen
    // bytes consecutivos dentro del bloque y tienen que quedar alineados a 16.
    const int CH_ = ((CHK / SCH + BK - 1) / BK) * BK;
    const int kini = kbase + trozo * CH_;
    int lim_ = kbase + (trozo + 1) * CH_;
    lim_ = lim_ < kbase + CHK ? lim_ : kbase + CHK;
    const int kfin = lim_ < N ? lim_ : N;
#else
    const int kini = kbase;
    const int kfin = (kini + CHK) < N ? (kini + CHK) : N;
#endif
    const size_t KOFF = (size_t)CHK * NH * QD;
    const size_t EOFF = 2 * KOFF;
    const int hlim = bq + BQ;
    // Pagina sin tokens de esta secuencia: m = MINIT, S = 0 (la union la ignora).
#if HIB
    // Trozo vacio (mas alla del final de la pagina o de la secuencia): ranura vacia y chau.
    // Sin esto el maximo MINIT pasaba por la conversion de unidades y la union lo tomaba como
    // una pagina real con O basura.
    if (kini >= kfin) {
        if (lane == 0 && warp == 0)
            for (int r = 0; r < BQ; ++r) {
                out_m[(size_t)ranura_out * R + bq + r] = MINIT;
                out_s[(size_t)ranura_out * R + bq + r] = 0;
                out_c[(size_t)ranura_out * R + bq + r] = 0;
            }
        return;
    }
#endif
    if (kini >= N) {
#if HIB
        if (lane == 0 && warp == 0)   // trozo sin tokens: ranura vacia
            for (int r = 0; r < BQ; ++r) {
                out_m[(size_t)ranura_out * R + bq + r] = MINIT;
                out_s[(size_t)ranura_out * R + bq + r] = 0;
                out_c[(size_t)ranura_out * R + bq + r] = 0;
            }
        return;
#endif
        if (lane == 0 && warp == 0)
            for (int r = 0; r < BQ; ++r) { out_m[(size_t)tramo * R + bq + r] = MINIT; out_s[(size_t)tramo * R + bq + r] = 0; }
        return;
    }
#if HIB
    // Espejo: solo las paginas cuya ranura sigue siendo de este bloque (si no, las hace el int4)
    const int ranura = sq_ * PAGS + (tramo % PAGS);
    if (dueno[ranura] != bt[(size_t)sq_ * BTS + tramo]) {
        if (lane == 0 && warp == 0)   // no espejada: la hace el int4, esta ranura queda vacia
            for (int r = 0; r < BQ; ++r) {
                out_m[(size_t)ranura_out * R + bq + r] = MINIT;
                out_s[(size_t)ranura_out * R + bq + r] = 0;
                out_c[(size_t)ranura_out * R + bq + r] = 0;
            }
        return;
    }
    const signed char* base = pool + (size_t)ranura * (size_t)BLK;
#else
    // Pagina fisica: UNA carga de la tabla de bloques.
    const signed char* base = pool + (size_t)bt[(size_t)sq_ * BTS + tramo] * (size_t)BLK;
#endif

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
#if ARBOL
    int lb[2], am[2];
#endif
#pragma unroll
    for (int e = 0; e < 2; ++e) {
        const int r = bq + wq + tig * 2 + e;
        const int vale = r < hlim;
        qr[e] = r;
        lm[e] = vale ? lim[r] : -1;
#if ARBOL
        lb[e] = vale ? abase[r] : -1;
        am[e] = vale ? amask[r] : 0;
#endif
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

#define SK18H_CARGA_K(ET, K0)                                                  \
    do {                                                                       \
        for (int v = tid; v < BK * CHQ; v += nthreads) {                       \
            const int r = v / CHQ, c = (v % CHQ) * 16;                         \
            const int key = (K0) + r;                                          \
            const int ok = key < kfin;                                         \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"     \
                         :: "r"(sdir(&sK[ET][r][swz(r, c, CHQ)])),             \
                            "l"(base + ((size_t)((ok ? key : (K0)) - kbase) * NH + hh) * QD + c), "r"(ok ? 16 : 0)); \
        }                                                                      \
        for (int v = tid; v < BK * NH * 4 / 16; v += nthreads) {               \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"     \
                         :: "r"(sdir(&sE[ET][v * 16])),                        \
                            "l"(base + EOFF + (size_t)((K0) - kbase) * NH * 4 + v * 16), "r"(16)); \
        }                                                                      \
        __pipeline_commit();                                                   \
    } while (0)

// Visibilidad por ancestros (ARBOL): sin ramas, y sin desplazamientos fuera de rango.
#if ARBOL
#define SK18H_ANCEXP(KEY, E)                                                   \
    (((KEY) <= lb[E]) | ((int)((unsigned)((KEY) - lb[E] - 1) < 31u) &          \
                         ((am[E] >> (((KEY) - lb[E] - 1) & 31)) & 1)))
// El chequeo NO va en el lazo de keys: ahi se evalua una vez por cada key del contexto, cuando
// solo puede cambiar algo en las ultimas L (los tokens del paso). Medido: tenerlo adentro cuesta
// 10-12% del kernel ENTERO, haya arbol o no (contexto 4k: 370,8 us sin el, 415,3 con el). Se
// aplica DESPUES, sobre zq, y solo en la pagina que alcanza a `lb`: es un lazo sobre registros,
// sin mma, asi que no duplica el cuerpo desenrollado (eso hacia que ptxas no terminara).
#define SK18H_POSANC(K0)                                                       \
    do {                                                                       \
        const int _lbmin = lb[0] < lb[1] ? lb[0] : lb[1];                      \
        if ((K0) + BK > _lbmin) {                                              \
            for (int f = 0; f < BK / 16; ++f)                                  \
                for (int hr = 0; hr < 2; ++hr)                                 \
                    for (int e = 0; e < 2; ++e) {                              \
                        const int key = (K0) + f * 16 + gid + hr * 8;          \
                        if (!SK18H_ANCEXP(key, e)) zq[f * 2 + hr][e] = PADZ;   \
                    }                                                          \
        }                                                                      \
    } while (0)
#else
#define SK18H_POSANC(K0)
#endif

// Q.K de una unidad (buf 0) -> zq[BK/8][2] con PAD donde la key no es visible.
#define SK18H_QK(ET, K0)                                                       \
    do {                                                                       \
        for (int f = 0; f < BK / 16; ++f) {                                    \
            int acc[4] = {0, 0, 0, 0};                                         \
            for (int d = 0; d < 8; ++d) {                                      \
                unsigned a[4];                                                 \
                const int ra = f * 16 + la_row;                                \
                unsigned pa = sdir(&sK[ET][ra][swz(ra, d * 32 + la_col, CHQ)]); \
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" \
                             : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(pa)); \
                asm volatile("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 " \
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n" \
                             : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]) \
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(qb[d][0]), "r"(qb[d][1])); \
            }                                                                  \
            for (int hr = 0; hr < 2; ++hr)                                     \
                for (int e = 0; e < 2; ++e) {                                  \
                    const int key = (K0) + f * 16 + gid + hr * 8;              \
                    const int vis = key < kfin && key <= lm[e];                 \
                    const int sk = vis ? (int)((short*)&sE[ET][((key - (K0)) * NH + hh) * 4])[0] : 0; \
                    const int zz = (int)(((long long)acc[hr * 2 + e] * sk) >> ZSH); \
                    zq[f * 2 + hr][e] = vis ? zz : PADZ;                       \
                }                                                              \
        }                                                                      \
    } while (0)

#define SK18H_CARGA(ET, K0)                                                    \
    do {                                                                       \
        for (int v = tid; v < BK * CHQ; v += nthreads) {                       \
            const int r = v / CHQ, c = (v % CHQ) * 16;                         \
            const int key = (K0) + r;                                          \
            const int ok = key < kfin;                                         \
            if (DIAG != 5)                                                     \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"     \
                         :: "r"(sdir(&sK[ET][r][swz(r, c, CHQ)])),             \
                            "l"(base + ((size_t)((ok ? key : (K0)) - kbase) * NH + hh) * QD + c), "r"(ok ? 16 : 0)); \
        }                                                                      \
        for (int v = tid; v < QD * CHV; v += nthreads) {                       \
            const int r = v / CHV, c = (v % CHV) * 16;                         \
            const int nk = kfin - ((K0) + c);                                  \
            const int sz = nk >= 16 ? 16 : (nk > 0 ? nk : 0);                  \
            if (DIAG != 4)                                                     \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"     \
                         :: "r"(sdir(&sV[ET][r][swz(r, c, CHV)])),             \
                            "l"(base + KOFF + ((size_t)hh * QD + r) * CHK + ((K0) - kbase) + c), "r"(sz)); \
        }                                                                      \
        for (int v = tid; v < BK * NH * 4 / 16; v += nthreads) {               \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"     \
                         :: "r"(sdir(&sE[ET][v * 16])),                        \
                            "l"(base + EOFF + (size_t)((K0) - kbase) * NH * 4 + v * 16), "r"(16)); \
        }                                                                      \
        __pipeline_commit();                                                   \
    } while (0)

    // ── Pasada A: maximo EXACTO de la pagina por query (solo K + escalas + Q.K). ──
    // Pasada B: pesos contra ese maximo ya fijo y w.v, sin tocar nunca los acumuladores del
    // mma con operaciones de lane (el reescalado en streaming leia/escribia registros que el
    // pipe tensorial todavia estaba acumulando: oh/ol cambiaban entre corridas identicas).
    // Pipeline de carga de 2 etapas: la unidad siguiente se copia mientras se calcula la actual.
    // (La carrera que se le atribuyo al pipeline era del reescalado en streaming.)
    int buf = 0;
    SK18H_CARGA_K(0, kini);
    for (int k0 = kini; k0 < kfin; k0 += BK) {
        int zq[BK / 8][2];
        if (k0 + BK < kfin) { SK18H_CARGA_K((buf + 1) % 2, k0 + BK); } else { __pipeline_commit(); }
        __pipeline_wait_prior(1);
        __syncthreads();
        SK18H_QK(buf, k0);
        SK18H_POSANC(k0);
#pragma unroll
        for (int e = 0; e < 2; ++e)
#pragma unroll
            for (int j = 0; j < BK / 8; ++j) m[e] = (zq[j][e] != PADZ && zq[j][e] > m[e]) ? zq[j][e] : m[e];
        // Barrera de cierre de unidad: las copias de la unidad siguiente se emiten ARRIBA del
        // lazo, antes de la espera, asi que sin esto pisan la etapa que otro warp sigue leyendo.
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

    buf = 0;
    SK18H_CARGA(0, kini);
    for (int k0 = kini; k0 < kfin; k0 += BK) {
        int zq[BK / 8][2];
        if (k0 + BK < kfin) { SK18H_CARGA((buf + 1) % 2, k0 + BK); } else { __pipeline_commit(); }
        __pipeline_wait_prior(1);
        __syncthreads();
        SK18H_QK(buf, k0);
        SK18H_POSANC(k0);
        // pesos -> W (fila = query del bloque, columna = key de la unidad)
#pragma unroll
        for (int j = 0; j < BK / 8; ++j)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int col = (j >> 1) * 16 + gid + (j & 1) * 8;
                const int key = k0 + col;
                const int pad = zq[j][e] == PADZ;
                int d = m[e] - zq[j][e];
                d = d < dc[e] ? d : dc[e];
                const int w0 = pot2neg_q15((d * mq[e] + 32768) >> 16);
                const int w = pad ? 0 : ((w0 * WCAPW) >> 15);
                const int sv = pad ? 0 : (int)((short*)&sE[buf][((key - k0) * NH + hh) * 4])[1];
                const int wp = (w * sv + (1 << (VSH - 1))) >> VSH;
                const int hi = (wp + 128) >> 8;
                const int row = wq + tig * 2 + e;
                S[e] += w;
                sW[row][swzB(row, 0, col & ~15, CHV) + (col & 15)] = (signed char)hi;
                sW[row][swzB(row, 1, col & ~15, CHV) + (col & 15)] = (signed char)(wp - (hi << 8));
            }
        __syncthreads();
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
            for (int i = 0; i < MT; ++i) {
                unsigned av[4];
                const int ra = i * 16 + la_row;
                unsigned pa = sdir(&sV[buf][ra][swz(ra, kk + la_col, CHV)]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(av[0]), "=r"(av[1]), "=r"(av[2]), "=r"(av[3]) : "r"(pa));
                asm volatile("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+r"(oh[i][0]), "+r"(oh[i][1]), "+r"(oh[i][2]), "+r"(oh[i][3])
                             : "r"(av[0]), "r"(av[1]), "r"(av[2]), "r"(av[3]), "r"(wh[0]), "r"(wh[1]));
                asm volatile("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+r"(ol[i][0]), "+r"(ol[i][1]), "+r"(ol[i][2]), "+r"(ol[i][3])
                             : "r"(av[0]), "r"(av[1]), "r"(av[2]), "r"(av[3]), "r"(wl[0]), "r"(wl[1]));
            }
        }
        __syncthreads();
        buf = (buf + 1) % 2;
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
#if HIB   // maximo en unidades de z del int4: m * mqb8 / mqb4
            const long long q4 = mqb4[r] > 0 ? mqb4[r] : 1;
            out_m[(size_t)ranura_out * R + r] = (int)(((long long)m[e] * mq[e] + q4 / 2) / q4);
            out_c[(size_t)ranura_out * R + r] = 0;
            out_s[(size_t)ranura_out * R + r] = S[e];
#else
            out_m[(size_t)tramo * R + r] = m[e];
            out_s[(size_t)tramo * R + r] = S[e];
#endif
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
#if HIB
                    size_t o = ((size_t)ranura_out * R + r) * QD + i * 16 + gid + hr * 8;
#else
                    size_t o = ((size_t)tramo * R + r) * QD + i * 16 + gid + hr * 8;
#endif
#if HIB   // el espejo usa una referencia 2^ESC8 mas chica: O baja ESC8 bits
                    const long long O = ((long long)oh[i][hr * 2 + e] * 256 + ol[i][hr * 2 + e]) >> ESC8;
                    const int hi2 = (int)(O >> 8);
                    out_hi[o] = hi2;
                    out_lo[o] = (int)(O - ((long long)hi2 << 8));
#else
                    out_hi[o] = oh[i][hr * 2 + e];
                    out_lo[o] = ol[i][hr * 2 + e];
#endif
                }
            }
}
