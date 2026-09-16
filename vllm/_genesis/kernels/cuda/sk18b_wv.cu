// SPDX-License-Identifier: Apache-2.0
//
// SK-18b — segunda pasada del decode de SK-18: softmax ENTERO + sum(w * v), sin
// punto flotante.
//
// Entradas (una cabeza KV):
//   Sz    int32 [M, K]   logits escalados z' = (q8.k8 >> 8) * skf (pasada 1 + escala
//                        de key en punto fijo); relleno = -(1 << 28)
//   zmax  int32 [M]      maximo de z' por query
//   mq    int32 [M]      multiplicador de distancia a indice: idx = (d * mq) >> 16
//   invS  int32 [M]      (sin uso: la normalizacion por sum(w) va afuera, en int64)
//   lut   int32 [1024]   32639 * exp(-c * i / 1023): peso de 15 bits RELATIVO AL MAXIMO
//   svf   int16 [K]      escala de V por key relativa al maximo (Q15)
//   Vt    int8  [N, K]   V en layout POR DIMENSION (N = 256): B del mma
//   out/out_lo int32 [NCH, M, N]  sum_k hi*Vt y sum_k lo*Vt por tramo (<= CH*127*127)
//
// Peso w' = (w * svf) >> 15, de 15 bits, partido en dos bytes con signo
// hi = (w' + 128) >> 8, lo = w' - 256*hi: sum(w'*v) = 256*sum(hi*v) + sum(lo*v).
// Los dos planos se calculan en la etapa de carga y se escriben en shared; el
// paso de mma es el de SK-12 con gate = hi y up = lo contra el MISMO tile de V.
//
// Keys en tramos de CH (grilla: tiles de query x tiles de dimension x tramos):
// en decode M es chico y K es el contexto, asi que el paralelismo va en los tramos.
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
// DIAG (perfil): 0 normal; 1 carga de peso trivial (sin tabla ni mult); 2 sin mma.
#ifndef DIAG
#define DIAG 0
#endif


// v9 — mma.m16n8k32 + ldmatrix a mano. Se sale de WMMA.
//
// Por que
// -------
// WMMA fija el fragmento en m16n16k16. En int8 la instruccion rapida de Ampere
// es mma.sync.aligned.m16n8k32, que consume el DOBLE de K por instruccion. Con
// WMMA emitiamos el doble de mma para la misma matematica, y ese es el gap
// estructural con cutlass que el tuning de tiles no podia cerrar (se probaron 6
// variantes, ver REGISTRO DE TUNING: el techo con WMMA fue 74,7 TOPS).
//
// Ademas, controlar el fragmento a mano habilita dos cosas que con WMMA eran
// imposibles:
//   * el epilogo sale DIRECTO de los registros del acumulador, sin pasar por
//     shared. Eso libera los 16 KB del buffer de epilogo y sube la ocupacion.
//   * el swizzle XOR del shared (el padding fallo justamente porque WMMA se
//     comia el camino vectorizado de ldmatrix; ver v3).
//
// Mapeo de fragmentos (PTX ISA, mma.m16n8k32.row.col.s32.s8.s8.s32)
// -----------------------------------------------------------------
// A (16x32 int8, 4 regs/hilo): con ldmatrix.x4 sobre el tile visto como
// 16x16 b16, el orden de matrices que devuelve ldmatrix coincide exactamente
// con lo que espera mma:
//     lane 0-7   -> filas 0-7,  bytes 0-15   = a0
//     lane 8-15  -> filas 8-15, bytes 0-15   = a1
//     lane 16-23 -> filas 0-7,  bytes 16-31  = a2
//     lane 24-31 -> filas 8-15, bytes 16-31  = a3
//
// B (32x8 int8, 2 regs/hilo): el peso ya esta en shared como [n][k], que ES
// col-major para mma. ldmatrix.x2 con lanes 0-7 -> bytes 0-15 y 8-15 -> 16-31.
//
// C (16x8 int32, 4 regs/hilo):
//     c0,c1 -> fila groupID,     col tid_in_group*2 + {0,1}
//     c2,c3 -> fila groupID + 8, idem
// Esas dos columnas son contiguas, asi que el store sale en half2.

#ifndef BM
#define BM 64
#endif
#ifndef BN
#define BN 64
#endif
// BK=128: duplica el reuso de cada cp.async y, sobre todo, da 8 chunks de 16 B
// por fila en vez de 4, que es lo que permite que el swizzle XOR elimine los
// bank conflicts del TODO en A (con 4 chunks y 8 lanes el minimo era 2 vias).
// Pide 64 KB de shared, por encima del limite estatico de 48: va como shared
// DINAMICA con opt-in via cuFuncSetAttribute.
#ifndef BK
#define BK 128
#endif
// Sobreescribibles por -D desde el lanzador (GENESIS_SK12_DEFS) para barrer
// configuraciones sin editar el fuente.
#ifndef NWARPS
#define NWARPS 4
#endif
#ifndef WM
#define WM 64
#endif
#ifndef WN
#define WN 16
#endif
// Alto (en tiles de BM) del parche de rasterizacion. 1 = orden natural.
#ifndef GROUP_M
#define GROUP_M 1
#endif
// Profundidad del pipeline de global->shared. shared = STAGES*(BM+2*BN)*BK.
// Con BM=256 BN=64 el limite de 99 KB de sm_86 da: BK=128 -> 2, BK=64 -> 4.
#ifndef STAGES
#define STAGES 2
#endif
// Hints de desalojo de L2 en los cp.async. DEFAULT 0: medido, no dan nada.
//   sin hints        1,04 GB de DRAM  L2 88,81%  9,325 ms
//   solo A protegido 1,04 GB          L2 88,79%  9,273 ms  <- igual, dentro del ruido
//   solo B desalojado 1,28 GB         L2 85,93%  9,751 ms  <- PEOR
// Proteger A no cambia nada porque ya se quedaba, y marcar B evict_first mata su
// reuso de CORTA distancia: los GROUP_M bloques que comparten un tile de B corren
// casi a la vez en la misma oleada. El swizzle ya extrajo lo que habia.
#ifndef L2HINT
#define L2HINT 0
#endif
#define MTILES (WM / 16)    // 2 fragmentos m16 por warp
#define NTILES (WN / 8)     // 4 fragmentos n8 por warp

// #5 swizzle XOR del shared. Con BK=64 cada fila son 4 chunks de 16 B. Sin
// permutar, los 8 lanes de ldmatrix.x4 leen filas separadas 64 B = 16 bancos:
// las filas 0,2,4,6 caen en los mismos 4 bancos -> conflicto de 4 vias. Con
// chunk ^= (fila & 3) queda de 2 vias, que es el minimo alcanzable con 4
// chunks y 8 filas. El padding (v3) no podia hacer esto porque WMMA elegia el
// patron de acceso; con ldmatrix a mano lo elegimos nosotros.
//
// La MISMA permutacion se aplica al escribir (cp.async) y al leer (ldmatrix),
// asi que el dato logico (fila, chunk) siempre vive en (fila, chunk ^ fila&3).
// #5 swizzle XOR. La MISMA permutacion al escribir (cp.async) y al leer
// (ldmatrix), asi el dato logico siempre vive en su posicion permutada.
//
// A: fila de 64 B = 4 chunks de 16. Los 8 lanes de ldmatrix.x4 leen filas
// separadas 64 B = 16 bancos -> conflicto de 4 vias. Con chunk ^= (fila & 3)
// queda en 2, que es el minimo con 4 chunks.
// ── Invariantes de configuracion ──────────────────────────────────────────
//
// Todo esto es barrible por -D, y una combinacion incoherente NO daba error:
// con BM=128 y NWARPS=8 la grilla de warps queda 2x2=4, los otros 4 warps
// escriben fuera del tile y el kernel devuelve basura a 0,5x de velocidad
// sin quejarse. Pasa una vez y se pierde media hora. Que falle al compilar.
static_assert(BM % WM == 0, "BM tiene que ser multiplo de WM");
static_assert(BN % WN == 0, "BN tiene que ser multiplo de WN");
static_assert((BM / WM) * (BN / WN) == NWARPS,
              "la grilla de warps (BM/WM)x(BN/WN) tiene que dar exactamente "
              "NWARPS; si no, sobran warps que escriben fuera del tile");
static_assert(BK % 32 == 0, "BK tiene que ser multiplo de 32 (k del mma)");
static_assert(BK == 64 || BK == 128,
              "el bucle interno esta desenrollado a mano solo para BK 64 o 128");
static_assert(STAGES >= 2, "el pipeline necesita al menos 2 etapas");
static_assert(STAGES * (2 * BM * BK + BN * 2 * BK) <= 101376,
              "no entra en la shared dinamica de sm_86 (99 KB)");
static_assert(WM % 16 == 0 && WN % 8 == 0,
              "el fragmento del mma es m16n8k32");

// Chunks de 16 B por fila. Con BK=128 son 8; con BK=64, 4.
#define SK12_CH (BK / 16)

__device__ __forceinline__ int swzA(int row, int byte_col) {
    // El XOR con (fila & (CH-1)) reparte los lanes de ldmatrix.x4 en los bancos.
    // Con CH=8 (BK=128) el periodo alcanza para los 8 lanes: conflicto CERO.
    // Con CH=4 (BK=64) sólo hay 4 valores distintos, asi que vuelven conflictos
    // de 2 vias. Es un costo REAL de bajar BK, no una regresion.
    return ((byte_col >> 4) ^ (row & (SK12_CH - 1))) << 4;
}

// B: al intercalar gate/up la fila mide 128 B = 8 chunks = EXACTAMENTE 32
// bancos, asi que los 8 lanes caian todos en el mismo (conflicto de 8 vias).
// Con 8 chunks el XOR tiene periodo suficiente para repartir los 8 lanes en
// los 32 bancos sin pisarse: conflicto CERO. Devuelve el offset de byte
// completo dentro de la fila (m y k van juntos en el mismo indice).
__device__ __forceinline__ int swzB(int row, int mat, int byte_col) {
    // 16 chunks por fila (gate 8 + up 8). El XOR toca solo los 3 bits bajos,
    // asi que gate se queda en 0..7 y up en 8..15; el bit del `mat` se preserva.
    // 2*CH chunks por fila (gate CH + up CH). El XOR toca solo los bits bajos,
    // asi que el bit de `mat` se preserva y gate/up no se mezclan.
    const int chunk = (mat * SK12_CH) | (byte_col >> 4);
    return (chunk ^ (row & (SK12_CH - 1))) << 4;
}

__device__ __forceinline__ unsigned sdir(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

extern "C" __global__ void __launch_bounds__(NWARPS * 32)
sk18b_wv(
    const int* __restrict__ Sz,
    const int* __restrict__ zmax,
    const int* __restrict__ mq,
    const int* __restrict__ invS,
    const int* __restrict__ lut,
    const short* __restrict__ svf,
    const signed char* __restrict__ Vt,
    int* __restrict__ out,
    int* __restrict__ out_lo,
    int M, int N, int K, int CH, int NCH)
{
    const signed char* Wg = Vt;
    const signed char* Wu = Vt;
    // Sin buffer de epilogo: el acumulador se desescala en registros.
    //
    // Shared DINAMICA. Con BK=128 son 2*BM*BK + 2*BN*2*BK = 65536 B, por encima
    // del limite estatico de 48 KB, asi que el lanzador pide el opt-in a 100 KB
    // con cuFuncSetAttribute y pasa el tamano en el lanzamiento.
    //
    // gate y up van INTERCALADOS por fila n: sB[n][0..BK) = gate,
    // sB[n][BK..2BK) = up. Asi un unico ldmatrix.x4 devuelve las 4 matrices que
    // necesitamos en vez de dos .x2: la mitad de instrucciones de carga.
    extern __shared__ signed char _smem[];
    signed char (*sA)[BM][BK] = (signed char (*)[BM][BK]) _smem;                 // byte alto
    signed char (*sL)[BM][BK] = (signed char (*)[BM][BK]) (_smem + STAGES * BM * BK); // byte bajo
    signed char (*sB)[BN][2 * BK] =
        (signed char (*)[BN][2 * BK]) (_smem + 2 * STAGES * BM * BK);

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    // Grilla de warps (BM/WM) x (BN/WN); N es la dimension rapida.
    const int wm   = (warp / (BN / WN)) * WM;
    const int wn   = (warp % (BN / WN)) * WN;

    // Swizzle de rasterizacion (GROUP_M).
    //
    // Con el orden natural (blockIdx.x = N, la rapida) los 272 bloques
    // consecutivos comparten el tile de A pero cada uno trae un tile de B
    // distinto: dentro de una oleada de 82 SMs no se reusa B ni una vez, y
    // medido a M=8192 eso son 17,1 GB de trafico en 13,1 ms = 1,3 TB/s, o sea
    // L2 saturada. Reordenando a parches de GROUP_M x (oleada/GROUP_M), una
    // oleada toca pocos tiles de A y pocos de B, y los dos se reusan desde L2.
    const int pid_n = blockIdx.x;
    const int pid_m = blockIdx.y / NCH;
    const int tramo = blockIdx.y % NCH;
    const int kini = tramo * CH;
    const int kfin = (kini + CH) < K ? (kini + CH) : K;
    const int bm = pid_m * BM;
    const int bn = pid_n * BN;

    int accG[MTILES][NTILES][4];
    int accU[MTILES][NTILES][4];
#pragma unroll
    for (int i = 0; i < MTILES; ++i)
#pragma unroll
        for (int j = 0; j < NTILES; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) { accG[i][j][e] = 0; accU[i][j][e] = 0; }

    // Direcciones de ldmatrix, constantes en todo el bucle.
    const int la_row = (lane & 7) + 8 * ((lane >> 3) & 1);
    const int la_col = 16 * (lane >> 4);
    const int lb_row = lane & 7;              // fila n dentro del tile de 8
    const int lb_mat = (lane >> 4) & 1;       // 0 = gate, 1 = up
    const int lb_col = 16 * ((lane >> 3) & 1);

    const int vpr      = BK / 16;
    const int nthreads = NWARPS * 32;
    const int last_m   = M - 1;
    const int last_n   = N - 1;

// cp.async con politica de desalojo de L2.
//
// Medido con ncu a M=7488: la L2 va al 88,9% de aciertos y DRAM mueve 1,03 GB
// contra un minimo teorico de 257 MB. Y el tiempo CORRELACIONA con eso (28% de
// spread entre GROUP_M=8 y GROUP_M=30), asi que ese 11% de fallos se paga.
//
// El reparto sale de la asimetria del swizzle: dentro de un grupo de GROUP_M
// filas, los MISMOS tiles de A se reusan en los 136 pasos de N, mientras que
// cada tile de B lo consumen GROUP_M bloques y no vuelve nunca. O sea:
//   A -> evict_last   (protegerlo)
//   B -> evict_first  (que se vaya y no desaloje a A)
//
// `src-size` emula el zfill de __pipeline_memcpy_async: los bytes que faltan
// hasta cp-size se rellenan con cero, que es como se manejan los bordes.
#define SK12_CPA(DST, SRC, SRCSZ, POL)                                         \
    asm volatile("cp.async.cg.shared.global.L2::cache_hint "                   \
                 "[%0], [%1], 16, %2, %3;\n"                                   \
                 :: "r"(DST), "l"(SRC), "r"(SRCSZ), "l"(POL))

#define SK12_CPA_PLANO(DST, SRC, SRCSZ)                                        \
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"             \
                 :: "r"(DST), "l"(SRC), "r"(SRCSZ))

#if L2HINT == 1
#define SK12_LOAD_A(DST, SRC, SZ) SK12_CPA(DST, SRC, SZ, _polA)
#define SK12_LOAD_B(DST, SRC, SZ) SK12_CPA(DST, SRC, SZ, _polB)
#elif L2HINT == 2
#define SK12_LOAD_A(DST, SRC, SZ) SK12_CPA(DST, SRC, SZ, _polA)
#define SK12_LOAD_B(DST, SRC, SZ) SK12_CPA_PLANO(DST, SRC, SZ)
#elif L2HINT == 3
#define SK12_LOAD_A(DST, SRC, SZ) SK12_CPA_PLANO(DST, SRC, SZ)
#define SK12_LOAD_B(DST, SRC, SZ) SK12_CPA(DST, SRC, SZ, _polB)
#else
#define SK12_LOAD_A(DST, SRC, SZ) SK12_CPA_PLANO(DST, SRC, SZ)
#define SK12_LOAD_B(DST, SRC, SZ) SK12_CPA_PLANO(DST, SRC, SZ)
#endif

#define SK12_CARGA(BUF, K0)                                                    \
    do {                                                                       \
        for (int v = tid; v < BM * vpr; v += nthreads) {                       \
            const int r = v / vpr, c = (v % vpr) * 16;                         \
            const int gm = bm + r;                                             \
            signed char* dh_ = &sA[BUF][r][swzA(r, c)];                        \
            signed char* dl_ = &sL[BUF][r][swzA(r, c)];                        \
            if (gm < M) {                                                      \
                const long long zm_ = zmax[gm];                                \
                const long long mq_ = mq[gm];                                  \
                const int* srow_ = Sz + (size_t)gm * K;                        \
                for (int e_ = 0; e_ < 16; ++e_) {                              \
                    const int kk_ = (K0) + c + e_;                             \
                    int w_ = 0;                                                \
                    if (DIAG == 1 && kk_ < kfin) w_ = srow_[kk_] & 0x3fff;     \
                    if (DIAG != 1 && kk_ < kfin) {                             \
                        long long ix_ = ((zm_ - (long long)srow_[kk_]) * mq_ + 32768) >> 16; \
                        if (ix_ > 1023) ix_ = 1023;                            \
                        const long long wv_ = (long long)lut[ix_];             \
                        w_ = (int)((wv_ * (long long)svf[kk_] + 16384) >> 15); \
                    }                                                          \
                    const int hi_ = (w_ + 128) >> 8;                           \
                    dh_[e_] = (signed char)hi_;                                \
                    dl_[e_] = (signed char)(w_ - (hi_ << 8));                  \
                }                                                              \
            } else {                                                           \
                for (int e_ = 0; e_ < 16; ++e_) { dh_[e_] = 0; dl_[e_] = 0; }  \
            }                                                                  \
        }                                                                      \
        for (int v = tid; v < BN * vpr; v += nthreads) {                       \
            const int r = v / vpr, c = (v % vpr) * 16;                         \
            const int gn = bn + r;                                             \
            const int cn = gn < N ? gn : last_n;                               \
            SK12_LOAD_B(sdir(&sB[BUF][r][swzB(r, 0, c)]),                      \
                Wg + (size_t)cn * K + (K0) + c, gn < N ? 16 : 0);              \
            SK12_LOAD_B(sdir(&sB[BUF][r][swzB(r, 1, c)]),                      \
                Wu + (size_t)cn * K + (K0) + c, gn < N ? 16 : 0);              \
        }                                                                      \
        __pipeline_commit();                                                   \
    } while (0)


// ── Intercalado 1:1, copiando lo que hace cutlass ──────────────────────────
//
// El SASS de cutlass alterna IMMA LDSM IMMA LDSM uno a uno, y mete ademas los
// LDGSTS y el calculo de direcciones entre medio. El nuestro emitia LDSMx5 y
// despues IMMAx8: lo escribi asi a proposito ("rafaga de mma sin loads en el
// medio"), y resulto ser LA diferencia. Un IMMA ocupa el pipe tensorial varios
// ciclos durante los cuales el scheduler puede emitir gratis en el LSU y en la
// ALU; agrupando, el primer mma espera a su ldmatrix sin nada que llene el
// hueco y los 8 mma seguidos no tienen nada que co-emitir.
//
// Sacarle `volatile` a los mma para que ptxas reordenara solo NO funciona
// (medido: el SASS sale identico). Hay que intercalarlo en el fuente, y con
// `volatile` PUESTO en ambos lados para que no vuelva a reagrupar.

#define SK12_LDA(KK, I, A)                                                     \
        {                                                                      \
            const int ra_ = wm + (I) * 16 + la_row;                            \
            unsigned pa_ = sdir(&sA[buf][ra_][swzA(ra_, (KK) + la_col)]);      \
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "           \
                "{%0,%1,%2,%3}, [%4];\n"                                       \
                : "=r"(A[I][0]), "=r"(A[I][1]), "=r"(A[I][2]), "=r"(A[I][3])   \
                : "r"(pa_));                                                   \
        }

#define SK12_LDL(KK, I, A)                                                     \
        {                                                                      \
            const int ra_ = wm + (I) * 16 + la_row;                            \
            unsigned pa_ = sdir(&sL[buf][ra_][swzA(ra_, (KK) + la_col)]);      \
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "           \
                "{%0,%1,%2,%3}, [%4];\n"                                       \
                : "=r"(A[I][0]), "=r"(A[I][1]), "=r"(A[I][2]), "=r"(A[I][3])   \
                : "r"(pa_));                                                   \
        }

#define SK12_LDB(KK, J, BG, BU)                                                \
        {                                                                      \
            const int rb_ = wn + (J) * 8 + lb_row;                             \
            unsigned pb_ = sdir(&sB[buf][rb_][swzB(rb_, lb_mat, (KK)+lb_col)]);\
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "           \
                "{%0,%1,%2,%3}, [%4];\n"                                       \
                : "=r"(BG[J][0]), "=r"(BG[J][1]),                              \
                  "=r"(BU[J][0]), "=r"(BU[J][1])                               \
                : "r"(pb_));                                                   \
        }

#define SK12_FRAG(KK, A, AL, BG, BU)                                           \
        { _Pragma("unroll")                                                    \
          for (int i = 0; i < MTILES; ++i) { SK12_LDA(KK, i, A) SK12_LDL(KK, i, AL) } \
          _Pragma("unroll")                                                    \
          for (int j = 0; j < NTILES; ++j) SK12_LDB(KK, j, BG, BU) }

#define SK12_MMA1(ACC, A, B)                                                   \
        asm volatile(                                                          \
          "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "         \
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"            \
          : "+r"(ACC[0]), "+r"(ACC[1]), "+r"(ACC[2]), "+r"(ACC[3])             \
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),                        \
            "r"(B[0]), "r"(B[1]));

// Un paso de k: 32 mma, con las 8 cargas del paso siguiente repartidas de a una
// entre los primeros 8. CARGAR=0 en el ultimo paso del tile.
#define SK12_PASO(KSIG, A, AL, BG, BU, NA, NAL, NBG, NBU, CARGAR)              \
        {                                                                      \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < MTILES; ++i) {                                 \
                _Pragma("unroll")                                              \
                for (int j = 0; j < NTILES; ++j) {                             \
                    const int t_ = i * NTILES + j;                             \
                    SK12_MMA1(accG[i][j], A[i], BG[j])                         \
                    if ((CARGAR) && t_ < MTILES) { SK12_LDA(KSIG, t_, NA) SK12_LDL(KSIG, t_, NAL) } \
                    SK12_MMA1(accU[i][j], AL[i], BU[j])                        \
                    if ((CARGAR) && t_ < NTILES) SK12_LDB(KSIG, t_, NBG, NBU)  \
                }                                                              \
            }                                                                  \
        }

#if L2HINT
    // Politicas creadas UNA vez por bloque: son dos registros de 64 bits.
    unsigned long long _polA, _polB;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;"
                 : "=l"(_polA));
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;"
                 : "=l"(_polB));
#endif

    // Prologo: se largan STAGES-1 etapas antes de entrar al bucle.
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        const int ks = kini + s * BK;
        if (ks < kfin) { SK12_CARGA(s, ks); } else { __pipeline_commit(); }
    }

    int buf = 0;

    for (int k0 = kini; k0 < kfin; k0 += BK) {
        // Se emite SIEMPRE un commit, aunque la carga quede fuera de rango, para
        // que la cantidad de grupos en vuelo sea constante y wait_prior(STAGES-1)
        // valga en todo el bucle, incluida la cola. Un grupo vacio completa solo.
        const int knext = k0 + (STAGES - 1) * BK;
        const int slot = (buf + STAGES - 1) % STAGES;
        if (knext < kfin) { SK12_CARGA(slot, knext); } else { __pipeline_commit(); }
        __pipeline_wait_prior(STAGES - 1);
        __syncthreads();

        // Software pipelining a mano, intercalado 1:1 como cutlass.
        //
        // Dos juegos de registros de fragmentos: se precarga k=0 y las 8 cargas
        // del paso k+1 se reparten de a una entre los 32 mma del paso k, en vez
        // de emitirse todas juntas antes de la rafaga.
        {
            unsigned a0[MTILES][4], l0[MTILES][4], bg0[NTILES][2], bu0[NTILES][2];
            unsigned a1[MTILES][4], l1[MTILES][4], bg1[NTILES][2], bu1[NTILES][2];
            SK12_FRAG(0, a0, l0, bg0, bu0)
            // Desenrollado a mano para alternar los dos juegos de registros sin
            // copias. Es #if, no un if: se elige al compilar y el SASS sale
            // igual de plano en los dos casos.
#if DIAG == 2
#elif BK == 128
            SK12_PASO(32, a0, l0, bg0, bu0, a1, l1, bg1, bu1, 1)
            SK12_PASO(64, a1, l1, bg1, bu1, a0, l0, bg0, bu0, 1)
            SK12_PASO(96, a0, l0, bg0, bu0, a1, l1, bg1, bu1, 1)
            SK12_PASO(0,  a1, l1, bg1, bu1, a0, l0, bg0, bu0, 0)
#elif BK == 64
            SK12_PASO(32, a0, l0, bg0, bu0, a1, l1, bg1, bu1, 1)
            SK12_PASO(0,  a1, l1, bg1, bu1, a0, l0, bg0, bu0, 0)
#else
#error "BK tiene que ser 64 o 128: el bucle interno esta desenrollado a mano"
#endif
        }
        __syncthreads();
        buf = (buf + 1) % STAGES;
    }
#undef SK12_CARGA

    // Epilogo. El bucle principal es 100% entero (int8 -> int32, ni un cast ni
    // un flotante). Aca hay que salir a punto flotante SI O SI porque SiLU es
    // trascendente y el acumulador int32 desborda fp16, pero se hace UNA vez
    // por elemento y con las instrucciones aproximadas.
    const int gid = lane >> 2;
    const int tig = lane & 3;

#pragma unroll
    for (int i = 0; i < MTILES; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int gm = bm + wm + i * 16 + gid + h * 8;
            if (gm >= M) continue;
            int* orow = out + (size_t)tramo * M * N + (size_t)gm * N;
            int* orow_lo = out_lo + (size_t)tramo * M * N + (size_t)gm * N;
#pragma unroll
            for (int j = 0; j < NTILES; ++j) {
                const int c0  = bn + wn + j * 8 + tig * 2;
                const int idx = h * 2;
                if (c0 < N)     { orow[c0] = accG[i][j][idx]; orow_lo[c0] = accU[i][j][idx]; }
                if (c0 + 1 < N) { orow[c0 + 1] = accG[i][j][idx + 1]; orow_lo[c0 + 1] = accU[i][j][idx + 1]; }
            }
        }
}
