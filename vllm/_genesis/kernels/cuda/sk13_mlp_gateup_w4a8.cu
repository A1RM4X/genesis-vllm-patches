// SPDX-License-Identifier: Apache-2.0
//
// SK-13 — MLP gate_up + SiLU en W4A8: pesos int4 SIN convertir, desempaquetados
// dentro del kernel. Derivado de SK-12 (copiado y modificado).
//
// Por que W4A8 y no W8A8
// ----------------------
// Convertir los pesos AWQ de int4 a int8 DUPLICA la memoria de pesos:
//     MLP en int4   6,2 GiB      MLP en int8  12,3 GiB
//     pesos totales 8,82 GiB  ->             14,9 GiB
//     + KV cache   11,35 GiB   =             26,25 GiB  >  23,56 disponibles
// No entra. El swap 1:1 de PN110 funciona para FP8->INT8 (mismo tamano), no
// para W4->INT8. Asi que desempaquetar en el kernel no es una optimizacion:
// es la unica opcion que entra en la placa.
//
// El desempaque es GRATIS (medido)
// --------------------------------
// Microbenchmark en esta misma 3090, registros puros:
//     mma solo                    0,75 ms   285,2 TOPS
//     mma + desempaque int4       3,03 ms   283,7 TOPS   (4x los mma)
// Cuatro veces los mma en 4,04 veces el tiempo: las 5 instrucciones de lop3 y
// shifts por fragmento se esconden enteras en la sombra del mma, que usa otro
// pipe de emision.
//
// (De paso: dp4a junto al mma BAJA el mma de 285 a 228 TOPS. mma y FP32
// comparten puertos en GA102, asi que el "tensor core de pobres" pierde. Lo
// medido le da la razon al hack de separar rafagas, no al de mezclarlas.)
//
// El layout empaquetado, elegido para no perder ldmatrix
// ------------------------------------------------------
// El problema conocido de W4A8 es que ldmatrix no deja los nibbles donde mma
// los quiere. Se resuelve eligiendo el empaquetado, no peleandolo:
//
//     byte i de la fila n  =  (w4[n][k=i] & 0xF) | (w4[n][k=i+16] << 4)
//
// con i = 0..15 dentro de cada bloque de 32 k. Entonces, tras un ldmatrix que
// entrega 4 bytes por hilo:
//     lo = P & 0x0F0F0F0F        -> k = tig*4 + 0..3    = fragmento b0
//     hi = (P >> 4) & 0x0F0F0F0F -> k = tig*4 + 16..19  = fragmento b1
// que es EXACTAMENTE el mapeo que pide mma.m16n8k32 para B. Dos lop3 y un
// shift, sin permutacion de indices ni shuffles.
//
// Escalas por grupo
// -----------------
// AWQ cuantiza por grupos de 128 a lo largo de K, asi que el acumulador entero
// solo es valido DENTRO de un grupo. Cada 128 k (= 2 tiles de BK=64) se vuelca:
//     facc += (float)iacc * escala_grupo ;  iacc = 0
// Por eso hay dos juegos de acumuladores y el tile por warp es mas chico que
// en SK-12.
//
// REGISTRO DE TUNING — RTX 3090 @220W, M=2048 K=5120 N=8704
// ----------------------------------------------------------
//   v1  8 warps, tile 64x16, 204 registros    92,0 TOPS   <= ACTUAL
//   v2  16 warps, tile 32x16, 122 registros   72,7 TOPS   DESCARTADO
//
// Comparado en la MISMA corrida contra SK-12 (W8A8): 92,0 vs 97,3 TOPS.
// Empatan; W4A8 queda 6% atras.
//
// v2 baja los registros de 204 a 122 como se buscaba, pero con 4 warps en N el
// fragmento de A se carga 4 veces en vez de 2. Es exactamente lo que paso en
// SK-12 (v12/v13): en este kernel el cuello es ancho de banda de shared, no
// ocupacion, y redundar A cuesta el doble que redundar B porque el ldmatrix.x4
// mueve 512 B contra 256 del x2.
//
// El precio de la cuantizacion por grupo son los acumuladores DOBLES (int32
// dentro del grupo + fp32 acumulando entre grupos) = 128 registros para un
// tile de 32 elementos por hilo. Es lo que deja el kernel en 1 bloque por SM.
//
// MEJORAS pendientes
// ------------------
//   1. Sacar el zero point del camino caliente:
//        sum_k a_k*(w_k - zp) = sum_k a_k*w_k - zp * sum_k a_k
//      El segundo termino es zp por la suma de fila de A, precalculable una vez
//      por (fila, grupo). Ahorra los dos __vadd4 por fragmento y deja el
//      operando de mma como el nibble crudo.
//   2. Acumulador fp32 en la mitad de registros usando half2 para el acumulado
//      entre grupos (el rango lo permite tras la escala).
//   3. Warp specialization.
//
// Ampere: mma.sync.aligned.m16n8k32.s32.s8.s8.s32, ldmatrix.sync.aligned.m8n8,
// cp.async.cg.shared.global, lop3.b32 para el desempaque.

#include <cuda_fp16.h>
#include <cuda_pipeline.h>

#define BM 128
#define BN 64
#define BK 64
#define NWARPS 8
#define WM 64
#define WN 16
#define MTILES (WM / 16)    // 4
#define NTILES (WN / 8)     // 2
#define GRP 128             // group_size de AWQ
#define BYTES_N (2 * (BK / 2))   // por fila n: gate 32 B + up 32 B

__device__ __forceinline__ int swzB(int row, int chunk) {
    return (chunk ^ (row & 3)) << 4;   // 4 chunks de 16 B por fila
}
__device__ __forceinline__ int swzA(int row, int byte_col) {
    return ((byte_col >> 4) ^ (row & 3)) << 4;
}
__device__ __forceinline__ unsigned sdir(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

extern "C" __global__ void __launch_bounds__(NWARPS * 32)
sk13_mlp_gateup_w4a8(
    const signed char* __restrict__ A,      // [M, K] int8
    const unsigned char* __restrict__ Wp,   // [N][2][K/2] empaquetado
    const float* __restrict__ sa,           // [M]
    const float* __restrict__ sgrp,         // [K/GRP][2][N] escala por grupo
    const signed char* __restrict__ zp,     // [K/GRP][2][N] zero point
    __half* __restrict__ out,               // [M, N]
    int M, int N, int K)
{
    __shared__ signed char  sA[2][BM][BK];
    __shared__ unsigned char sB[2][BN][BYTES_N];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = (warp >> 2) * WM;   // 2 warps en M
    const int wn   = (warp & 3) * WN;    // 4 warps en N
    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;

    int   iacc[MTILES][NTILES][4], iaccU[MTILES][NTILES][4];
    float facc[MTILES][NTILES][4], faccU[MTILES][NTILES][4];
#pragma unroll
    for (int i = 0; i < MTILES; ++i)
#pragma unroll
        for (int j = 0; j < NTILES; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                iacc[i][j][e] = 0; iaccU[i][j][e] = 0;
                facc[i][j][e] = 0.f; faccU[i][j][e] = 0.f;
            }

    const int la_row = (lane & 7) + 8 * ((lane >> 3) & 1);
    const int la_col = 16 * (lane >> 4);
    const int lb_row = lane & 7;
    const int lb_mat = (lane >> 3) & 1;     // 0 = gate, 1 = up
    const int nthreads = NWARPS * 32;
    const int last_m = M - 1, last_n = N - 1;
    const int kb2 = K / 2;

#define SK13_CARGA(BUF, K0)                                                    \
    do {                                                                       \
        for (int v = tid; v < BM * (BK / 16); v += nthreads) {                 \
            const int r = v / (BK / 16), c = (v % (BK / 16)) * 16;             \
            const int gm = bm + r, cm = gm < M ? gm : last_m;                  \
            __pipeline_memcpy_async(&sA[BUF][r][swzA(r, c)],                   \
                A + (size_t)cm * K + (K0) + c, 16, gm < M ? 0 : 16);           \
        }                                                                      \
        for (int v = tid; v < BN * 4; v += nthreads) {                         \
            const int r = v >> 2, ch = v & 3;                                  \
            const int m = ch >> 1, kbk = ch & 1;                               \
            const int gn = bn + r, cn = gn < N ? gn : last_n;                  \
            __pipeline_memcpy_async(&sB[BUF][r][swzB(r, ch)],                  \
                Wp + ((size_t)cn * 2 + m) * kb2 + ((K0) >> 1) + kbk * 16,      \
                16, gn < N ? 0 : 16);                                          \
        }                                                                      \
        __pipeline_commit();                                                   \
    } while (0)

    SK13_CARGA(0, 0);
    int buf = 0;

    for (int k0 = 0; k0 < K; k0 += BK) {
        const int k1 = k0 + BK;
        if (k1 < K) SK13_CARGA(buf ^ 1, k1);
        __pipeline_wait_prior(k1 < K ? 1 : 0);
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; kk += 32) {
            const int kbk = kk >> 5;              // 0 o 1
            unsigned a[MTILES][4];
#pragma unroll
            for (int i = 0; i < MTILES; ++i) {
                const int ra = wm + i * 16 + la_row;
                unsigned p = sdir(&sA[buf][ra][swzA(ra, kk + la_col)]);
                asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                    : "=r"(a[i][0]), "=r"(a[i][1]), "=r"(a[i][2]), "=r"(a[i][3]) : "r"(p));
            }
#pragma unroll
            for (int j = 0; j < NTILES; ++j) {
                const int rb = wn + j * 8 + lb_row;
                // Un ldmatrix.x2 trae el registro empaquetado de gate Y de up.
                unsigned pb = sdir(&sB[buf][rb][swzB(rb, (lb_mat << 1) | kbk)]);
                unsigned pg, pu;
                asm("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                    : "=r"(pg), "=r"(pu) : "r"(pb));
                // Desempaque: 2 lop3 + 1 shift por matriz. El zero point se
                // resta por byte con __vsub4 para que no propague borrow.
                const int grp = k0 / GRP;
                const int cn0 = bn + wn + j * 8 + (lane >> 2);
                const int cnz = cn0 < N ? cn0 : last_n;
                const unsigned zg = __vsub4(0u, (unsigned)(unsigned char)
                    zp[((size_t)grp * 2 + 0) * N + cnz] * 0x01010101u);
                const unsigned zu = __vsub4(0u, (unsigned)(unsigned char)
                    zp[((size_t)grp * 2 + 1) * N + cnz] * 0x01010101u);
                unsigned gl, gh, ul, uh;
                asm("lop3.b32 %0, %1, 0x0F0F0F0F, 0, 0xC0;\n" : "=r"(gl) : "r"(pg));
                asm("shr.b32 %0, %1, 4;\n" : "=r"(gh) : "r"(pg));
                asm("lop3.b32 %0, %0, 0x0F0F0F0F, 0, 0xC0;\n" : "+r"(gh));
                asm("lop3.b32 %0, %1, 0x0F0F0F0F, 0, 0xC0;\n" : "=r"(ul) : "r"(pu));
                asm("shr.b32 %0, %1, 4;\n" : "=r"(uh) : "r"(pu));
                asm("lop3.b32 %0, %0, 0x0F0F0F0F, 0, 0xC0;\n" : "+r"(uh));
                gl = __vadd4(gl, zg); gh = __vadd4(gh, zg);
                ul = __vadd4(ul, zu); uh = __vadd4(uh, zu);
#pragma unroll
                for (int i = 0; i < MTILES; ++i) {
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                        : "+r"(iacc[i][j][0]), "+r"(iacc[i][j][1]),
                          "+r"(iacc[i][j][2]), "+r"(iacc[i][j][3])
                        : "r"(a[i][0]), "r"(a[i][1]), "r"(a[i][2]), "r"(a[i][3]),
                          "r"(gl), "r"(gh));
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                        : "+r"(iaccU[i][j][0]), "+r"(iaccU[i][j][1]),
                          "+r"(iaccU[i][j][2]), "+r"(iaccU[i][j][3])
                        : "r"(a[i][0]), "r"(a[i][1]), "r"(a[i][2]), "r"(a[i][3]),
                          "r"(ul), "r"(uh));
                }
            }
        }
        __syncthreads();
        buf ^= 1;

        // Frontera de grupo: el acumulador entero solo vale dentro del grupo.
        if ((k1 % GRP) == 0 || k1 >= K) {
            const int grp = k0 / GRP;
#pragma unroll
            for (int j = 0; j < NTILES; ++j) {
                const int cn0 = bn + wn + j * 8 + (lane >> 2);
                const int cnz = cn0 < N ? cn0 : last_n;
                const float eg = sgrp[((size_t)grp * 2 + 0) * N + cnz];
                const float eu = sgrp[((size_t)grp * 2 + 1) * N + cnz];
#pragma unroll
                for (int i = 0; i < MTILES; ++i)
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        facc[i][j][e]  += (float)iacc[i][j][e]  * eg;
                        faccU[i][j][e] += (float)iaccU[i][j][e] * eu;
                        iacc[i][j][e] = 0; iaccU[i][j][e] = 0;
                    }
            }
        }
    }
#undef SK13_CARGA

    const int gid = lane >> 2, tig = lane & 3;
    const bool par = ((N & 1) == 0);
#pragma unroll
    for (int i = 0; i < MTILES; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int gm = bm + wm + i * 16 + gid + h * 8;
            if (gm >= M) continue;
            const float as = sa[gm];
            __half* orow = out + (size_t)gm * N;
#pragma unroll
            for (int j = 0; j < NTILES; ++j) {
                const int c0 = bn + wn + j * 8 + tig * 2, idx = h * 2;
                const float g0 = facc[i][j][idx]      * as;
                const float g1 = facc[i][j][idx + 1]  * as;
                const float u0 = faccU[i][j][idx]     * as;
                const float u1 = faccU[i][j][idx + 1] * as;
                const float s0 = __fdividef(g0, 1.0f + __expf(-g0));
                const float s1 = __fdividef(g1, 1.0f + __expf(-g1));
                if (par && c0 + 1 < N)
                    *reinterpret_cast<__half2*>(orow + c0) =
                        __floats2half2_rn(s0 * u0, s1 * u1);
                else {
                    if (c0 < N)     orow[c0]     = __float2half(s0 * u0);
                    if (c0 + 1 < N) orow[c0 + 1] = __float2half(s1 * u1);
                }
            }
        }
}
