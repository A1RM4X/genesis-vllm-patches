// SPDX-License-Identifier: Apache-2.0
//
// SK-12 — MLP gate_up + SiLU fusionado, DIMENSIONADO PARA PREFILL.
//
// Por que existe
// --------------
// Los SK-01..SK-11 se compilaron para los M de decode (los .ptx congelados en
// assets/ptx_sk son M16/M20/M32/M40, que son las tallas de cudagraph). No hay
// ninguno para prefill, donde M vale 2048-8192. Un GEMM tuneado para M=16 y uno
// para M=8192 no se parecen en nada: cambian el tile, el reparto de warps y el
// patron de reuso.
//
// Que hace
// --------
//     out[m,n] = silu(A[m,:] . Wg[n,:] * sa[m] * sg[n])
//              * (A[m,:] . Wu[n,:] * sa[m] * su[n])
//
// en UNA pasada. El intermedio gate_up (M x 2N int32) nunca toca memoria
// global: es el 100% del trafico que se ahorra contra hacer GEMM + activacion
// por separado.
//
// Por que INT8 y no fp16
// ----------------------
// Medido en la 3090 de este equipo, forma real del MLP (M=2048 K=5120 N=17408):
//     fp16 cuBLAS        63,7 TFLOPS
//     INT8 cutlass      132,5 TOPS      -> 2,08x
// Los tensor cores enteros de GA102 corren al doble del rate de fp16, y el
// prefill es compute-bound, asi que ese 2x se traduce directo. En decode NO
// sirve: ahi el cuello es DRAM leyendo los pesos.
//
// Ampere: emite mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 via
// WMMA, y ldmatrix para llevar los fragmentos de shared a registros. El
// cp.async NO esta todavia: es la primera mejora obvia (ver MEJORAS abajo).
//
// Layout que espera
// -----------------
//   A   int8  [M, K]   row-major, cuantizado per-token
//   Wg  int8  [N, K]   row-major (gate)   = weight[0:N]   de gate_up_proj
//   Wu  int8  [N, K]   row-major (up)     = weight[N:2N]  de gate_up_proj
//   sa  fp32  [M]      escala per-token de A
//   sg  fp32  [N]      escala per-canal de Wg
//   su  fp32  [N]      escala per-canal de Wu
//   out fp16  [M, N]
//
// No hay permutacion de pesos. P113 permuta el peso en la carga y eso lo dejo
// atrapado: una vez permutado no se puede volver a cutlass para los M donde
// cutlass gana. Aca el peso se lee tal como esta, asi que el dispatch por M es
// reversible en cualquier momento.
//
// REGISTRO DE TUNING — RTX 3090 @220W, M=2048 K=5120 N=8704
// ----------------------------------------------------------
//   v1   WMMA ingenuo, BM=BN=64, load sincrono       55,7 TOPS   6,55 ms
//   v2   + cp.async y doble buffer                   59,6 TOPS   6,13 ms
//   v3   + padding del stride (BK+16)                53,4 TOPS   DESCARTADO
//   v4   BM=128 BN=64 BK=64, 8 warps, union shared   74,7 TOPS   4,89 ms
//   v5   BM=BN=128 BK=32, 16 warps                   62,7 TOPS   DESCARTADO
//   v6   BM=BN=128 BK=64, 16 warps, 48 KB shared     76,3 TOPS   empata
//   v7   v4 + __launch_bounds__(256, 3)              59,8 TOPS   DESCARTADO
//   v8   pipeline de 3 etapas                        45,6 TOPS   DESCARTADO
//   v9   mma.m16n8k32 + ldmatrix, epilogo en regs    84,2 TOPS   4,34 ms
//   v12  16 warps, WN=16 (acumulador 64->32 regs)    73,9 TOPS   DESCARTADO
//   v13  WM=64 WN=16                                 79,9 TOPS   DESCARTADO
//   v14  4 warps gordos, WM=64 WN=32                 85,0 TOPS   4,30 ms
//   v15  + gate/up intercalados (un x4 en vez de     86,6 TOPS   4,22 ms
//        dos x2: mitad de instrucciones de carga)
//   v16  + swizzle XOR en A (4->2 vias) y B (8->0)   94,2 TOPS   3,88 ms
//   v17  + ldmatrix sin 'volatile'                   96,3 TOPS   3,79 ms
//   v18  epilogo: __fdividef, direccion de fila en     104,7 TOPS  3,49 ms
//        64 bits UNA vez, escalas izadas, store half2
//   v19  + nvcc --use_fast_math                        117,6 TOPS  3,10 ms
//   v20  BK=128 con shared dinamica (64 KB, opt-in a    ver abajo
//        100 KB via cuFuncSetAttribute) + swizzle de A
//        con 8 chunks, que ahi si da conflicto CERO
//
// A partir de v20 el kernel quedo clavado en 0,81x cutlass y NINGUNA variante
// de emision lo movio. Lo que finalmente lo destrabo fue mirar el trafico, no
// las instrucciones:
//
//   v21  intercalado 1:1 de ldmatrix y mma en el       0,81x  SIN CAMBIO
//        fuente, copiando el SASS de cutlass. El PTX
//        sale intercalado pero ptxas lo RE-AGRUPA al
//        bajar a SASS: el orden del fuente no es
//        palanca en sm_86.
//   v22  8 warps en vez de 4 (2 por scheduler en vez   0,81x  SIN CAMBIO
//        de 1). Tampoco. Descarta que sea emision.
//   v23  BM 128 -> 256 (BN=64, 8 warps, 96 KB shared)  1,07x  GANA
//        Aca estaba: con BM=128 el grid es 272x16 y
//        cada bloque lee BM*K + 2*BN*K, o sea 5,7 GB
//        de trafico para operandos que pesan 100 MB.
//        Duplicar BM parte el grid al medio.
//        BM=128/BN=128 (mismo shared) da solo 0,92x:
//        B cuenta doble porque son gate Y up.
//   v24  + swizzle de rasterizacion GROUP_M=8          1,22x  a M=8192
//        Sin el, 272 bloques consecutivos comparten A
//        pero ninguno comparte B: una oleada de 82 SMs
//        no reusa B ni una vez. Medido a M=8192 eran
//        17,1 GB en 13,1 ms = 1,3 TB/s, o sea L2 al
//        palo. GROUP_M=4/8/16 dan 1,21/1,22/1,17x.
//
//   v25  hints de desalojo de L2 en cp.async         SIN CAMBIO / PEOR
//        Medido con ncu a M=7488: L2 88,9%, DRAM 1,03 GB
//        contra un minimo teorico de 257 MB. Parecia
//        que sobraba margen, pero no se puede cobrar:
//        A ya se queda en L2 y desalojar B mata su
//        reuso inmediato dentro de la oleada.
//
// CUANTO IMPORTA LA CACHE (medido, tiempo real sin profiler, M=7488):
//   GROUP_M=1   DRAM 2,91 GB  L2 65,9%  11,203 ms
//   GROUP_M=4   DRAM 1,14 GB  L2 87,6%   9,450 ms
//   GROUP_M=8   DRAM 1,03 GB  L2 88,9%   9,229 ms  <- optimo
//   GROUP_M=16  DRAM 1,43 GB  L2 84,1%   9,682 ms
//   GROUP_M=30  DRAM 3,58 GB  L2 57,7%  11,833 ms
// 28% de spread y correlaciona con el hit rate: la cache importa MUCHO, pero ya
// estamos en el optimo. OJO: el tiempo que reporta ncu NO sirve para esto (da
// 8,55-8,62 ms para todas las configs porque serializa y vacia cachas).
//
//   v26  pipeline profundo (BK=64 + STAGES 2/3/4)    PEOR / SIN CAMBIO
//        BK=128 2et  10,221 ms  1,06x   <- sigue ganando
//        BK=64  2et  11,598 ms  0,93x
//        BK=64  3et  11,746 ms  0,92x
//        BK=64  4et  11,781 ms  0,93x
//        Aislando SOLO la profundidad (BM=128, BK=128, swizzle optimo):
//        2 etapas 11,651 ms vs 3 etapas 11,625 ms = 0,2%. NADA.
//        BK=64 cuesta 13% aparte: con 4 chunks por fila el XOR del swizzle
//        pierde periodo y vuelven conflictos de banco de 2 vias.
//
// ══ POR QUE NADA DE ESTO FUNCIONO ══════════════════════════════════════════
// ncu, desglose de stalls a M=7488 (ciclos por emision, sobre 11,18):
//        math_pipe_throttle  5,30   <- 47%, el pipe TENSORIAL saturado
//        wait                2,43
//        barrier             0,76
//        not_selected        0,49
//        mio_throttle        0,45
//        long_scoreboard     0,31   <- memoria global: IRRELEVANTE
//        short_scoreboard    0,14
//
// El kernel esta limitado por el pipe tensorial: emite mma tan rapido como
// sm_86 los retira. Por eso fallaron TODOS los intentos de esta linea —
// intercalado 1:1, ocupancia, hints de L2, profundidad de pipeline: ninguno
// ataca el pipe tensorial. Lo unico que queda seria emitir menos mma, y la
// matematica es la que es (ya estamos en m16n8k32, el mas ancho de INT8 en
// Ampere). SK-12 esta terminado.
//
// Config final: BM=256 BN=64 BK=128, 8 warps WM=64 WN=32, GROUP_M=8.
// BM*BN=16384 es el TECHO DURO: el acumulador son WM*WN*2/32 = 128 registros
// por hilo y con 212 totales no entran 512 hilos en el banco de 65536. Para
// crecer el tile hay que bajar el acumulador primero.
//
// vs cutlass INT8+SiLU por talla de M (minimo de 15 rondas intercaladas):
//   M=256  0,93x | M=512  0,83x | M=1024 0,96x
//   M=2048 0,75x | M=4096 1,02x | M=8192 1,22x  (129,6 TOPS)
// El pico aparente a M=1024 (124 TOPS) es artefacto del benchmark: A mide
// 5,2 MB y entra en la L2 de 6 MB, asi que se reusa entre iteraciones.
// En M=2048 cutlass llega a 126 TOPS con una config propia; nosotros estamos
// planos en 99-111 y ganamos donde el cae.
//
// Todo se puede barrer sin editar el fuente:
//   GENESIS_SK12_DEFS="-DBM=256 -DBN=64 -DNWARPS=8 -DWM=64 -DWN=32 -DGROUP_M=8"
//
// MEDICION LIMPIA (9 rondas INTERCALADAS de 60 it., mediana, 8 s de
// calentamiento, GPU libre). Intercalar importa: con el cap de 220 W cada
// kernel deja la placa en otro estado termico y medir A-luego-B contamina B.
//
//     Marlin W4A16 + SiLU (hoy)   6,820 ms    53,5 TOPS   1,00x
//     cutlass INT8 + SiLU         3,278 ms   111,4 TOPS   2,08x
//     SK-12 BK=128                4,021 ms    90,8 TOPS   1,70x  = 0,82x cutlass
//
// Dispersion: SK-12 2,1%, cutlass 5,8%, Marlin 5,9%.
//
// OJO con medir con el servidor de vLLM prendido: en esas condiciones SK-12
// marcaba 149,8 TOPS, que es un ARTEFACTO del estado de potencia. Siempre con
// la GPU libre e intercalado.
//
// CONTRA QUE PERDEMOS, Y QUE SE PROBO (todo medido, GPU libre, 9 rondas
// intercaladas de 60 it., mediana, 8 s de calentamiento previo)
// ----------------------------------------------------------------------
// Config real de cutlass para esta forma, sacada con el profiler:
//     ThreadblockShape 128x128x64   WarpShape 64x64x64
//     InstructionShape 16x8x32      Stages 5   Sm80
// Nuestra geometria YA es la misma: BN=64 x 2 matrices = su BN=128, warp
// 64x32 x 2 = su 64x64, 4 warps, misma instruccion, mismo trafico por k.
//
//     variante                          ms      vs cutlass
//     BK=64, 2 etapas                  4,35       0,76x
//     BK=128, 2 etapas                 4,02       0,82x
//     BK=128, 3 etapas                 4,00       0,81x
//     geometria identica (BK=64, 5et)  4,45       0,72x
//     epilogo f16x2                    3,94       0,82x
//     f16x2 + h2rcp                    4,01       0,81x
//     software pipelining a mano       4,00       0,82x
//
// Se clava en 0,82x. Lo unico que movio la aguja fue BK 64->128 (+8 puntos).
//
// El SASS dice que le GANAMOS en todo lo medible (por IMMA dinamico: LDSM
// 0,25 contra 0,375; LDGSTS 0,56 contra 0,625; STS 0 contra 1,12; BAR 0,02
// contra 0,41; IMAD 1,66 contra 9,38) y perdemos igual. La diferencia esta en
// COMO cutlass intercala su bucle interno, no en que instrucciones emite.
//
// TRAMPA EN LA QUE CAI, para que no la repitan: comparar conteos ESTATICOS del
// SASS. El epilogo corre UNA vez y el bucle de mma 40 veces (K/BK). Los 832
// FMUL del epilogo parecian 6,5 por IMMA y son 0,16. Normalizar por frecuencia
// de ejecucion ANTES de concluir.
//
// Desglose del bloque completo:
//     cutlass GEMM solo     3,021 ms
//     SiLU*mul separado     0,374 ms   (11% de su camino)
//     cutlass + SiLU        3,389 ms
//     SK-12 fusionado       4,054 ms
// GEMM contra GEMM somos 1,34x mas lentos; la fusion vale 11%; neto 1,20x
// atras. Para ganar hay que meter el GEMM dentro del 11%.
//
// CONCLUSION OPERATIVA: hoy conviene cutlass (PN118 solo, sin PN119). Contra
// produccion cutlass es 2,09x y SK-12 1,68x. SK-12 queda documentado y
// validado por si algun dia cutlass no aplica.
//
// MEJORAS pendientes, en orden de valor esperado
// ----------------------------------------------
//   1. BK=128. Cuadruplica los pasos de k por tile (4 en vez de 2) y mejora el
//      reuso de cada cp.async. Pide shared dinamica con opt-in a 100 KB via
//      cuFuncSetAttribute (el _Nativo de sk05 ya sabe hacerlo).
//   2. Warp specialization (#26): dedicar 1 de los 4 warps a cargar con
//      bar.arrive y 3 a multiplicar.
//   3. Doble buffer EXPLICITO de fragmentos en registros, desenrollando los 2
//      pasos de k a mano. Quitar 'volatile' dio +2%; hacerlo a mano deberia dar
//      mas, a costa de ~64 registros mas.
//
// Se compila con nvcc a PTX y se congela; el runtime lo ensambla con ptxas y lo
// lanza por libcuda. No hay Triton en ningun punto de la cadena.

#include <cuda_fp16.h>
#include <cuda_pipeline.h>


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
#define BM 256
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
#define NWARPS 8
#endif
#ifndef WM
#define WM 64
#endif
#ifndef WN
#define WN 32
#endif
// Alto (en tiles de BM) del parche de rasterizacion. 1 = orden natural.
#ifndef GROUP_M
#define GROUP_M 8
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
static_assert(STAGES * (BM * BK + BN * 2 * BK) <= 101376,
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
sk12_mlp_gateup_prefill(
    const signed char* __restrict__ A,
    const signed char* __restrict__ Wg,
    const signed char* __restrict__ Wu,
    const float* __restrict__ sa,
    const float* __restrict__ sg,
    const float* __restrict__ su,
    __half* __restrict__ out,
    int M, int N, int K)
{
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
    signed char (*sA)[BM][BK] = (signed char (*)[BM][BK]) _smem;
    signed char (*sB)[BN][2 * BK] =
        (signed char (*)[BN][2 * BK]) (_smem + STAGES * BM * BK);

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
    const int pid_lin = blockIdx.x + gridDim.x * blockIdx.y;
    const int n_m = gridDim.y, n_n = gridDim.x;
    const int por_grupo = GROUP_M * n_n;
    const int grupo = pid_lin / por_grupo;
    const int prim_m = grupo * GROUP_M;
    const int alto = (n_m - prim_m) < GROUP_M ? (n_m - prim_m) : GROUP_M;
    const int pid_m = prim_m + (pid_lin % alto);
    const int pid_n = (pid_lin % por_grupo) / alto;

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
            const int cm = gm < M ? gm : last_m;                               \
            SK12_LOAD_A(sdir(&sA[BUF][r][swzA(r, c)]),                         \
                A + (size_t)cm * K + (K0) + c, gm < M ? 16 : 0);               \
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

#define SK12_FRAG(KK, A, BG, BU)                                               \
        { _Pragma("unroll")                                                    \
          for (int i = 0; i < MTILES; ++i) SK12_LDA(KK, i, A)                  \
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
#define SK12_PASO(KSIG, A, BG, BU, NA, NBG, NBU, CARGAR)                       \
        {                                                                      \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < MTILES; ++i) {                                 \
                _Pragma("unroll")                                              \
                for (int j = 0; j < NTILES; ++j) {                             \
                    const int t_ = i * NTILES + j;                             \
                    SK12_MMA1(accG[i][j], A[i], BG[j])                         \
                    if ((CARGAR) && t_ < MTILES) SK12_LDA(KSIG, t_, NA)        \
                    SK12_MMA1(accU[i][j], A[i], BU[j])                         \
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
        const int ks = s * BK;
        if (ks < K) { SK12_CARGA(s, ks); } else { __pipeline_commit(); }
    }

    int buf = 0;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Se emite SIEMPRE un commit, aunque la carga quede fuera de rango, para
        // que la cantidad de grupos en vuelo sea constante y wait_prior(STAGES-1)
        // valga en todo el bucle, incluida la cola. Un grupo vacio completa solo.
        const int knext = k0 + (STAGES - 1) * BK;
        const int slot = (buf + STAGES - 1) % STAGES;
        if (knext < K) { SK12_CARGA(slot, knext); } else { __pipeline_commit(); }
        __pipeline_wait_prior(STAGES - 1);
        __syncthreads();

        // Software pipelining a mano, intercalado 1:1 como cutlass.
        //
        // Dos juegos de registros de fragmentos: se precarga k=0 y las 8 cargas
        // del paso k+1 se reparten de a una entre los 32 mma del paso k, en vez
        // de emitirse todas juntas antes de la rafaga.
        {
            unsigned a0[MTILES][4], bg0[NTILES][2], bu0[NTILES][2];
            unsigned a1[MTILES][4], bg1[NTILES][2], bu1[NTILES][2];
            SK12_FRAG(0, a0, bg0, bu0)
            // Desenrollado a mano para alternar los dos juegos de registros sin
            // copias. Es #if, no un if: se elige al compilar y el SASS sale
            // igual de plano en los dos casos.
#if BK == 128
            SK12_PASO(32, a0, bg0, bu0, a1, bg1, bu1, 1)
            SK12_PASO(64, a1, bg1, bu1, a0, bg0, bu0, 1)
            SK12_PASO(96, a0, bg0, bu0, a1, bg1, bu1, 1)
            SK12_PASO(0,  a1, bg1, bu1, a0, bg0, bu0, 0)
#elif BK == 64
            SK12_PASO(32, a0, bg0, bu0, a1, bg1, bu1, 1)
            SK12_PASO(0,  a1, bg1, bu1, a0, bg0, bu0, 0)
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

    // Escalas de COLUMNA: dependen de j y del lane, no de i. Antes se leian
    // dentro del bucle de i, o sea MTILES veces de mas cada una.
    float sgv[NTILES][2], suv[NTILES][2];
#pragma unroll
    for (int j = 0; j < NTILES; ++j) {
        const int c0 = bn + wn + j * 8 + tig * 2;
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int gn = c0 + e;
            sgv[j][e] = gn < N ? sg[gn] : 0.0f;
            suv[j][e] = gn < N ? su[gn] : 0.0f;
        }
    }

    // El store en half2 pide la fila alineada a 4 B. c0 siempre es par, asi
    // que alcanza con que N lo sea. Es uniforme en todo el warp.
    const bool par = ((N & 1) == 0);

#pragma unroll
    for (int i = 0; i < MTILES; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int gm = bm + wm + i * 16 + gid + h * 8;
            if (gm >= M) continue;
            // Escala de FILA: una sola lectura, fuera del bucle de j.
            const float as = sa[gm];
            // #32 direccionamiento: la base de la fila se calcula UNA vez en 64
            // bits y el resto indexa en 32. Antes cada elemento pagaba
            // mul.wide.s32 + add.s64 por su (size_t)gm * N + gn.
            __half* orow = out + (size_t)gm * N;
#pragma unroll
            for (int j = 0; j < NTILES; ++j) {
                const int c0  = bn + wn + j * 8 + tig * 2;
                const int idx = h * 2;
                const float asg0 = as * sgv[j][0], asg1 = as * sgv[j][1];
                const float asu0 = as * suv[j][0], asu1 = as * suv[j][1];
                // #21 f16x2: las dos columnas del acumulador son contiguas,
                // asi que todo el SiLU va vectorizado — DOS elementos por
                // instruccion. El SASS decia que el epilogo en fp32 era el
                // UNICO eje donde perdiamos contra cutlass (6,50 FMUL por IMMA
                // contra sus 2,25), y en GA102 las unidades FP32 comparten
                // puertos de emision con los tensor cores (#10): medido, meter
                // trabajo FP32 junto al mma baja los IMMA de 285 a 228 TOPS.
                //
                // La desescalada se queda en fp32 a proposito: el acumulador
                // llega a ~1e8 y desbordaria fp16. Se convierte DESPUES de
                // escalar, cuando el valor ya ronda 1e3-1e4 y entra holgado en
                // los 65504 de fp16.
                const __half2 g = __floats2half2_rn(
                    (float)accG[i][j][idx]     * asg0,
                    (float)accG[i][j][idx + 1] * asg1);
                const __half2 u = __floats2half2_rn(
                    (float)accU[i][j][idx]     * asu0,
                    (float)accU[i][j][idx + 1] * asu1);
                // silu(g) = g / (1 + exp(-g)), todo en f16x2:
                // h2exp -> ex2.approx.f16x2, una instruccion para los dos.
                // __h2div NO es una instruccion: el compilador la expande en
                // Newton-Raphson y mete cientos de HADD2/F2FP. h2rcp baja a
                // MUFU directo, y la division se vuelve una multiplicacion.
                const __half2 sil = __hmul2(
                    g, h2rcp(__hadd2(__float2half2_rn(1.0f), h2exp(__hneg2(g)))));
                const __half2 res = __hmul2(sil, u);
                if (par && c0 + 1 < N) {
                    // Las dos columnas son contiguas: un store de 32 bits.
                    *reinterpret_cast<__half2*>(orow + c0) = res;
                } else {
                    if (c0 < N)     orow[c0]     = __low2half(res);
                    if (c0 + 1 < N) orow[c0 + 1] = __high2half(res);
                }
            }
        }
}
