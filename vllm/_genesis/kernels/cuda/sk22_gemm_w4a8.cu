// SK-22 — GEMM W4A8 propio para M chico (el decode), con PTX donde importa.
//
// Por que existe
// --------------
// El Marlin de PN130 esta muy bien pero paga una maquinaria que el decode no necesita: reparto
// stream-K entre bloques, locks en global, buffer C_tmp para la reduccion, y un tile de 16 filas
// del que con M=5 se aprovechan 5. Todo eso deja el lazo caliente en 103 registros y 230
// IMAD.MOV.U32 — movimientos hechos en la ALU entera porque no quedan puertos de MOV libres — y
// por eso PN140 (el nibble crudo de QServe) salio MAS LENTO ahi: no habia lugar para meter una
// lectura de shared y una resta.
//
// Aca el reparto es trivial: un bloque por tile de N, cada uno recorre todo K. Con N/TN bloques
// (136 para gate_up, 970 para el lm_head) sobra trabajo para los 82 SMs, asi que no hace falta
// stream-K ni locks ni C_tmp. Eso libera los registros que PN140 necesitaba.
//
// Que hace
// --------
//   C[M,N] = sum_k A[M,k] * (B4[k,N] - 8) * s[k/G, N]      A int8, B4 uint4, s int16 con signo
//
// El offset de 8 va por el camino de QServe: el nibble entra CRUDO al tensor core y se corrige
// con 8 * sum_k a_k, que llega precalculado por (grupo, fila). Es algebra exacta — el desempaque
// queda en `and` y `shr` en vez de las siete instrucciones del |MASK - zp ^MASK.
//
// Layout
// ------
//   A      [M, K]        int8, filas contiguas
//   B      [K/32, (N/8)*32]  int32, en el orden que consume el fragmento del mma (MEDIDO en
//                            tests/proto/sk22_layout.py): para el grupo de 8 columnas g y el
//                            lane L, el entero B[ktile][g*32 + L] lleva en su byte j el nibble
//                            bajo de q[k=(L%4)*4+j][n=g*8+L/4] y el alto de q[k+16][n].
//   esc    [K/G, N]      int16 con signo (PN130), G = 128
//   sumas  [K/G, M]      int32, sum_k a_k por grupo — avanzar un grupo es sumar M
//   C      [M, N]        fp16
//
// Geometria: 4 warps, TN=64 columnas por bloque, 16 filas de M (una sola m_block).

#include <cuda_fp16.h>
#include <cstdint>

#ifndef TN
  #define TN 64            // columnas de C por bloque
#endif
#ifndef ETAPAS
  #define ETAPAS 3         // profundidad del pipeline de cp.async sobre B
#endif
#define G 128              // group_size de las escalas
#ifndef WARPS
  #define WARPS 4
#endif
#define HILOS (WARPS * 32)
#define KT 32              // lo que consume UN mma.m16n8k32 — fijo, es la instruccion
#ifndef KPI
  #define KPI 128          // K por iteracion del lazo: KPI/KT mma seguidos por warp
#endif
#define NKT (KPI / KT)     // tiles de mma por iteracion
static_assert(KPI % KT == 0, "KPI tiene que ser multiplo de 32");
static_assert(KPI <= G, "una iteracion no puede cruzar dos grupos de escalas");

// ── helpers PTX ──────────────────────────────────────────────────────────────────────────────

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// cp.async de 16 bytes, el ancho maximo. Es lo que hace Marlin y esta bien: el perfil confirmo
// 41 cp.async.cg de 16 B, o sea que por ahi no se pierde nada.
__device__ __forceinline__ void carga16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}

// Variante con src-size: si `bytes` es 0, cp.async rellena el destino de ceros sin tocar
// memoria. Sirve para las filas f >= M del tile de A, que tienen que quedar en cero, sin
// necesidad de un camino aparte ni de un __syncthreads() extra para limpiarlas.
__device__ __forceinline__ void carga16p(uint32_t dst, const void* src, bool valido) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               ::"r"(dst), "l"(src), "r"(valido ? 16 : 0));
}

__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n"); }

template <int N>
__device__ __forceinline__ void esperar() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// mma entero: 16x8x32 sobre int8, acumulando en int32. Consume K=32 por instruccion, el doble
// que el HMMA fp16 — es la razon de trabajar en enteros y no solo la precision.
__device__ __forceinline__ void mma_s8(const uint32_t a[4], const uint32_t b[2], int32_t c[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}

// El swizzle XOR de CUTLASS, en trozos de 16 B. Hace falta porque shA tiene filas de KPI
// bytes: con KPI=128 eso son 32 bancos justos, asi que TODAS las filas arrancan en el banco 0
// y una lectura por filas choca de a 8 o de a 16 vias. Con el XOR, la fila f corre sus trozos
// f lugares, y los 32 lanes caen en bancos distintos. El mismo XOR va en la escritura.
#define TROZOS (KPI / 16)
__device__ __forceinline__ int swz(int f, int c) {
  return (c & ~15) ^ ((f & (TROZOS - 1)) * 16) | (c & 15);
}

__device__ __forceinline__ void ldmatrix4(uint32_t r[4], uint32_t dir) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(dir));
}

// ── el kernel ────────────────────────────────────────────────────────────────────────────────

extern "C" __global__ __launch_bounds__(HILOS) void sk22_gemm_w4a8(
    const int8_t* __restrict__ A,       // [M, K]
    const int32_t* __restrict__ B,      // [K/16, N*8] nibbles empaquetados
    const int16_t* __restrict__ esc,    // [K/G, N] escalas int16 con signo
    const int32_t* __restrict__ sumas,  // [K/G, M] sum_k a_k por grupo
    const float* __restrict__ a_esc,    // [M] escala de la activacion por fila
    __half* __restrict__ C,             // [M, N]
    int M, int N, int K, float factor) {
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;

  // shared: A del tile (16 x KT) y B de las etapas del pipeline
  extern __shared__ char sh[];
  // A tambien va por el pipeline: recargarla sincronicamente en cada vuelta costaba el 40% del
  // kernel (medido: 90 -> 53,5 us en 5120x5120), porque el ld.global + __syncthreads() de cada
  // iteracion no tiene con que solaparse. Es chica (512 B por etapa), asi que entra de sobra.
  constexpr int A_POR_ETAPA = 16 * KPI;
  int8_t* shA = reinterpret_cast<int8_t*>(sh);                   // ETAPAS * A_POR_ETAPA
  int32_t* shB = reinterpret_cast<int32_t*>(sh + ETAPAS * A_POR_ETAPA);

  // B por etapa: un tile de K (32 filas) x TN columnas = TN/8 grupos de 8, y cada grupo son 32
  // enteros (uno por lane). Son 256 int32 = 1 KiB con TN=64.
  constexpr int GRUPOS = TN / 8;
  constexpr int B_POR_TILE = GRUPOS * 32;          // un tile de mma: 32 filas de K x TN columnas
  constexpr int B_POR_ETAPA = NKT * B_POR_TILE;
  constexpr int B_VEC = B_POR_TILE / 4;            // cargas de 16 B por tile

  // DOS acumuladores, como hace Marlin: el del mma se resetea al cerrar cada grupo de K, y el
  // escalado va juntando. Hace falta porque la escala y la correccion cambian POR GRUPO — con un
  // solo acumulador sobre todo K, el resultado solo es correcto si hay un unico grupo.
  const int ntiles = (N + TN - 1) / TN;
  for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
  const int n0 = tile * TN;
  int32_t acc[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};        // el del mma, por grupo
  float acc_esc[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};      // ya escalado, sobre todo K
  const int col = n0 + warp * 16;   // 16 columnas por warp = 2 grupos de 8 del mma

  // ── prologo del pipeline ───────────────────────────────────────────────────────────────────
  const size_t base_b = (size_t)tile * (K / KT) * B_POR_TILE;   // tramo de este tile

  // Trae a la etapa `e` el bloque de K que empieza en `k`. B no es contiguo entre tiles de mma
  // (cada uno salta ngrupos_n*32), asi que se carga tile por tile; adentro de cada tile si.
  auto traer = [&](int e, int k) {
    if (k >= K) return;
  #pragma unroll
    for (unsigned i = tid; i < NKT * B_VEC; i += HILOS) {
      unsigned kt = i / B_VEC, j = (i % B_VEC) * 4;
      // los tiles que caen pasado K se rellenan de cero (src-size 0), asi un K que no es
      // multiplo de KPI no lee fuera de rango ni necesita un lazo de cola aparte
      bool ok = k + (int)kt * KT < K;
      carga16p(smem_u32(&shB[e * B_POR_ETAPA + kt * B_POR_TILE + j]),
               &B[base_b + (size_t)(k / KT + (ok ? (int)kt : 0)) * B_POR_TILE + j], ok);
    }
    constexpr int A_VEC = KPI / 16;            // cargas de 16 B por fila de A
  #pragma unroll
    for (unsigned i = tid; i < 16u * A_VEC; i += HILOS) {
      unsigned f = i / A_VEC, c = (i % A_VEC) * 16;
      bool ok = (int)f < M && k + (int)c < K;
      carga16p(smem_u32(&shA[e * A_POR_ETAPA + f * KPI + swz(f, c)]),
               &A[(size_t)f * K + k + (ok ? (int)c : 0)], ok);
    }
  };

  __syncthreads();   // el tile anterior todavia podia estar leyendo shared
  for (int e = 0; e < ETAPAS - 1; e++) {
    traer(e, e * KPI);
    commit();
  }

  const int ngrupos = K / G;
  int grupo_prev = -1;
  // DOS correcciones, no una: los cuatro acumuladores del fragmento cubren las filas lane/4
  // (g=0,1) y lane/4+8 (g=2,3), y cada una tiene su propia suma de A.
  int32_t corr[2] = {0, 0};
  int32_t s_col[2][2] = {{0, 0}, {0, 0}};   // [grupo j del mma][columna par/impar]

  // la fila de C que ve este lane, con el layout del mma m16n8k32
  const int fila = lane / 4;

  int etapa = 0, etapa_n = ETAPAS - 1;
  for (int k = 0; k < K; k += KPI) {

    esperar<ETAPAS - 2>();
    __syncthreads();

    // escalas y correccion: solo cuando cambia el grupo, no en cada iteracion
    const int grupo = k / G;
    if (grupo != grupo_prev) {
      // cerrar el grupo anterior con SUS escalas antes de pisarlas
      if (grupo_prev >= 0) {
  #pragma unroll
        for (int j = 0; j < 2; j++)
  #pragma unroll
          for (int g = 0; g < 4; g++) {
            acc_esc[j][g] += float(acc[j][g] - corr[g / 2]) * float(s_col[j][g % 2]);
            acc[j][g] = 0;
          }
      }
      grupo_prev = grupo;
  #pragma unroll
      for (int j = 0; j < 2; j++) {
        int cb = col + j * 8 + (lane % 4) * 2;
        s_col[j][0] = (cb < N) ? esc[grupo * N + cb] : 0;
        s_col[j][1] = (cb + 1 < N) ? esc[grupo * N + cb + 1] : 0;
      }
  #pragma unroll
      for (int h = 0; h < 2; h++) {
        int f = lane / 4 + h * 8;
        corr[h] = 8 * ((sumas && f < M) ? sumas[grupo * M + f] : 0);
      }
    }

    // A -> registros del mma, con el layout medido:
    //   fila = lane/4 + (reg%2)*8      k = (lane%4)*4 + byte + (reg/2)*16
    // Se arma leyendo de shared directo. ldmatrix pide un orden distinto y para 16 filas no
    // compensa el reacomodo.
  #pragma unroll
    for (int kt = 0; kt < NKT; kt++) {
      // A -> registros con UNA instruccion. Medido en sk22_sondal.cu: tratando A int8 [16][32]
      // como b16 [16][16], el fragmento de mma.m16n8k32.s8 es exactamente la convencion de las
      // cuatro matrices 8x8 de ldmatrix.x4 (reg 0: filas 0-7 cols 0-7; reg 1: filas 8-15; regs
      // 2 y 3 idem con las columnas altas). Cero diferencias en las 128 posiciones.
      uint32_t fa[4];
      {
        int f = lane % 16, c = kt * KT + (lane / 16) * 16;
        ldmatrix4(fa, smem_u32(&shA[etapa * A_POR_ETAPA + f * KPI + swz(f, c)]));
      }

    // B: desempaque CRUDO (QServe) — dos instrucciones, sin el |MASK -zp ^MASK.
    // El layout esta MEDIDO (tests/proto/sk22_layout.py, 256 posiciones, cero discrepancias):
    // el lane L lee un solo int32 del grupo de 8 columnas g, y sus nibbles bajos son el
    // registro 0 del fragmento (k = (L%4)*4 + byte) y los altos el registro 1 (k + 16).
  #pragma unroll
      for (int j = 0; j < 2; j++) {
        int g = warp * 2 + j;                     // grupo de 8 columnas dentro del tile
        int32_t emp = shB[etapa * B_POR_ETAPA + kt * B_POR_TILE + g * 32 + lane];
        uint32_t fb[2];
        fb[0] = emp & 0x0F0F0F0F;
        fb[1] = (emp >> 4) & 0x0F0F0F0F;
        mma_s8(fa, fb, acc[j]);
      }
    }

    // siguiente etapa
    int kn = k + (ETAPAS - 1) * KPI;
    traer(etapa_n, kn);
    commit();
    if (++etapa == ETAPAS) etapa = 0;
    if (++etapa_n == ETAPAS) etapa_n = 0;
    __syncthreads();
  }

  // ── epilogo: correccion del offset, escalas y salida ───────────────────────────────────────
  // La correccion se resta ANTES de multiplicar por la escala: el termino que sobra es ~17x el
  // resultado, y multiplicarlo primero haria crecer el intermedio sin necesidad.
  {
    // cerrar el ultimo grupo, que quedo abierto al salir del lazo
  #pragma unroll
    for (int j = 0; j < 2; j++)
  #pragma unroll
      for (int g = 0; g < 4; g++)
        acc_esc[j][g] += float(acc[j][g] - corr[g / 2]) * float(s_col[j][g % 2]);

    // la escala de activacion tambien es por fila: una para lane/4 y otra para lane/4+8
    float ea2[2];
  #pragma unroll
    for (int h = 0; h < 2; h++) {
      int f = lane / 4 + h * 8;
      ea2[h] = (f < M ? a_esc[f] : 0.0f) * factor;
    }
  #pragma unroll
    for (int j = 0; j < 2; j++) {
  #pragma unroll
      for (int g = 0; g < 4; g++) {
        // C de m16n8: el lane tiene filas lane/4 y +8, columnas (lane%4)*2 y +1
        int c = col + j * 8 + (lane % 4) * 2 + (g % 2);
        int f = lane / 4 + (g / 2) * 8;
        if (f < M && c < N) {
          float v = acc_esc[j][g] * ea2[g / 2];
          C[f * N + c] = __float2half(v);
        }
      }
    }
  }
  }
}
