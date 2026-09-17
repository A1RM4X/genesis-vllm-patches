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
//   B      [K/16, N*8]   int32, el mismo empaquetado que ya usa Marlin (8 nibbles por int32)
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
  int8_t* shA = reinterpret_cast<int8_t*>(sh);                   // 16 * KT
  int32_t* shB = reinterpret_cast<int32_t*>(sh + 16 * KT);       // ETAPAS * (KT/16 * TN*8/8)

  // B por etapa: KT filas de 16 => KT/16 bloques de int32, cada uno TN*8/8 = TN enteros
  constexpr int B_POR_ETAPA = (KT / 16) * TN;

  int32_t acc[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};   // 2 mma de n8 por warp => 16 columnas
  const int col = n0 + warp * 16;                      // columnas que toca este warp

  // ── prologo del pipeline ───────────────────────────────────────────────────────────────────
  for (int e = 0; e < ETAPAS - 1; e++) {
    int k = e * KT;
    if (k < K && tid < B_POR_ETAPA / 4) {
      carga16(smem_u32(&shB[e * B_POR_ETAPA + tid * 4]),
              &B[(k / 16) * (N * 8) + (n0 * 8) + tid * 4]);
    }
    commit();
  }

  const int ngrupos = K / G;
  int grupo_prev = -1;
  int32_t corr = 0;          // 8 * sum_k a_k de la fila que le toca a este lane
  int16_t s_col[2] = {0, 0};

  // la fila de C que ve este lane, con el layout del mma m16n8k32
  const int fila = lane / 4;

  for (int k = 0; k < K; k += KT) {
    const int etapa = (k / KT) % ETAPAS;

    // A del tile: lo cargan todos los hilos, es chico (16 x 32 = 512 B)
    if (tid < 16 * KT / 4) {
      int f = (tid * 4) / KT, c = (tid * 4) % KT;
      const int8_t* src = &A[f * K + k + c];
      int32_t v = (f < M && k + c < K) ? *reinterpret_cast<const int32_t*>(src) : 0;
      *reinterpret_cast<int32_t*>(&shA[f * KT + c]) = v;
    }
    esperar<ETAPAS - 2>();
    __syncthreads();

    // escalas y correccion: solo cuando cambia el grupo, no en cada iteracion
    const int grupo = k / G;
    if (grupo != grupo_prev) {
      grupo_prev = grupo;
      s_col[0] = esc[grupo * N + col + (lane % 4) * 2];
      s_col[1] = esc[grupo * N + col + (lane % 4) * 2 + 1];
      corr = 8 * (sumas ? sumas[grupo * M + (fila < M ? fila : 0)] : 0);
    }

    // A -> registros del mma
    uint32_t fa[4];
    ldmatrix4(fa, smem_u32(&shA[(lane % 16) * KT + (lane / 16) * 16]));

    // B: desempaque CRUDO (QServe) — dos instrucciones, sin el |MASK -zp ^MASK
  #pragma unroll
    for (int j = 0; j < 2; j++) {
      int32_t emp = shB[etapa * B_POR_ETAPA + (warp * 16 + j * 8) + (lane % 8)];
      uint32_t fb[2];
      fb[0] = emp & 0x0F0F0F0F;
      fb[1] = (emp >> 4) & 0x0F0F0F0F;
      mma_s8(fa, fb, acc[j]);
    }

    // siguiente etapa
    int kn = k + (ETAPAS - 1) * KT;
    if (kn < K && tid < B_POR_ETAPA / 4) {
      carga16(smem_u32(&shB[((kn / KT) % ETAPAS) * B_POR_ETAPA + tid * 4]),
              &B[(kn / 16) * (N * 8) + (n0 * 8) + tid * 4]);
    }
    commit();
    __syncthreads();
  }

  // ── epilogo: correccion del offset, escalas y salida ───────────────────────────────────────
  // La correccion se resta ANTES de multiplicar por la escala: el termino que sobra es ~17x el
  // resultado, y multiplicarlo primero haria crecer el intermedio sin necesidad.
  if (fila < M) {
    const float ea = a_esc[fila] * factor;
  #pragma unroll
    for (int j = 0; j < 2; j++) {
  #pragma unroll
      for (int g = 0; g < 4; g++) {
        int c = col + j * 8 + (lane % 4) * 2 + (g % 2);
        int f = fila + (g / 2) * 8;
        if (f < M && c < N) {
          float v = float(acc[j][g] - corr) * float(s_col[g % 2]) * ea;
          C[f * N + c] = __float2half(v);
        }
      }
    }
  }
}
