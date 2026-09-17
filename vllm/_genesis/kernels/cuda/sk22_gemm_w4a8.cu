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
#define WARPS 4
#define HILOS (WARPS * 32)
#define KT 32              // K por iteracion: lo que consume un mma.m16n8k32

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
  const int n0 = blockIdx.x * TN;        // primera columna de este bloque

  // shared: A del tile (16 x KT) y B de las etapas del pipeline
  extern __shared__ char sh[];
  // A tambien va por el pipeline: recargarla sincronicamente en cada vuelta costaba el 40% del
  // kernel (medido: 90 -> 53,5 us en 5120x5120), porque el ld.global + __syncthreads() de cada
  // iteracion no tiene con que solaparse. Es chica (512 B por etapa), asi que entra de sobra.
  constexpr int A_POR_ETAPA = 16 * KT;                           // 512 B
  int8_t* shA = reinterpret_cast<int8_t*>(sh);                   // ETAPAS * A_POR_ETAPA
  int32_t* shB = reinterpret_cast<int32_t*>(sh + ETAPAS * A_POR_ETAPA);

  // B por etapa: un tile de K (32 filas) x TN columnas = TN/8 grupos de 8, y cada grupo son 32
  // enteros (uno por lane). Son 256 int32 = 1 KiB con TN=64.
  constexpr int GRUPOS = TN / 8;
  constexpr int B_POR_ETAPA = GRUPOS * 32;

  // DOS acumuladores, como hace Marlin: el del mma se resetea al cerrar cada grupo de K, y el
  // escalado va juntando. Hace falta porque la escala y la correccion cambian POR GRUPO — con un
  // solo acumulador sobre todo K, el resultado solo es correcto si hay un unico grupo.
  int32_t acc[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};        // el del mma, por grupo
  float acc_esc[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};      // ya escalado, sobre todo K
  const int col = n0 + warp * 16;   // 16 columnas por warp = 2 grupos de 8 del mma

  // ── prologo del pipeline ───────────────────────────────────────────────────────────────────
  const int ngrupos_n = N / 8;                 // grupos de 8 columnas de toda la matriz
  const int g0 = n0 / 8;                       // primer grupo de este bloque
  for (int e = 0; e < ETAPAS - 1; e++) {
    int kt = e;
    if (kt * KT < K) {
      if (tid < B_POR_ETAPA / 4)
        carga16(smem_u32(&shB[e * B_POR_ETAPA + tid * 4]),
                &B[(size_t)kt * ngrupos_n * 32 + g0 * 32 + tid * 4]);
      if (tid < A_POR_ETAPA / 16) {
        int f = tid / (KT / 16), c = (tid % (KT / 16)) * 16;
        carga16p(smem_u32(&shA[e * A_POR_ETAPA + f * KT + c]),
                 &A[(size_t)f * K + kt * KT + c], f < M);
      }
    }
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

  for (int k = 0; k < K; k += KT) {
    const int etapa = (k / KT) % ETAPAS;

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
    uint32_t fa[4];
  #pragma unroll
    for (int r = 0; r < 4; r++) {
      int f = lane / 4 + (r % 2) * 8;
      int kk = (lane % 4) * 4 + (r / 2) * 16;
      fa[r] = *reinterpret_cast<const uint32_t*>(&shA[etapa * A_POR_ETAPA + f * KT + kk]);
    }

    // B: desempaque CRUDO (QServe) — dos instrucciones, sin el |MASK -zp ^MASK.
    // El layout esta MEDIDO (tests/proto/sk22_layout.py, 256 posiciones, cero discrepancias):
    // el lane L lee un solo int32 del grupo de 8 columnas g, y sus nibbles bajos son el
    // registro 0 del fragmento (k = (L%4)*4 + byte) y los altos el registro 1 (k + 16).
  #pragma unroll
    for (int j = 0; j < 2; j++) {
      int g = warp * 2 + j;                       // grupo de 8 columnas dentro del tile
      int32_t emp = shB[etapa * B_POR_ETAPA + g * 32 + lane];
      uint32_t fb[2];
      fb[0] = emp & 0x0F0F0F0F;
      fb[1] = (emp >> 4) & 0x0F0F0F0F;
      mma_s8(fa, fb, acc[j]);
    }

    // siguiente etapa
    int kn = k + (ETAPAS - 1) * KT;
    if (kn < K) {
      const int en = (kn / KT) % ETAPAS;
      if (tid < B_POR_ETAPA / 4)
        carga16(smem_u32(&shB[en * B_POR_ETAPA + tid * 4]),
                &B[(size_t)(kn / KT) * ngrupos_n * 32 + g0 * 32 + tid * 4]);
      if (tid < A_POR_ETAPA / 16) {
        int f = tid / (KT / 16), c = (tid % (KT / 16)) * 16;
        carga16p(smem_u32(&shA[en * A_POR_ETAPA + f * KT + c]),
                 &A[(size_t)f * K + kn + c], f < M);
      }
    }
    commit();
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
