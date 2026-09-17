// Banco del desempaque int4 -> int8, aislado de todo lo demas.
//
// La pregunta: LiquidQuant (arXiv 2509.01229) dice dequantizar 4 elementos con 2 instrucciones
// (IMAD + XOR) fusionando la escala; Marlin usa 4 (and/or/sub/xor) y escala despues del mma.
// Leyendo el codigo parecia lo mismo — los dos usan el truco del ^0x80 — pero no lo es: Marlin
// NO mete la escala adentro. Esto lo mide en vez de discutirlo.
//
// Se mide solo la aritmetica: los datos entran de un buffer chico que vive en L2, para que el
// ancho de banda no tape la diferencia. Lo que importa es el costo por instruccion emitida.

#include <cstdio>
#include <cuda_runtime.h>

#define REP 2048

// ── 1) lo que hace Marlin hoy ────────────────────────────────────────────────────────────────
__device__ __forceinline__ void dequant_marlin(int q, int* out) {
  constexpr int repeated_zp = 0x08080808;
  constexpr int MASK = 0x80808080;
  out[0] = ((q & 0x0F0F0F0F | MASK) - repeated_zp) ^ MASK;
  q >>= 4;
  out[1] = ((q & 0x0F0F0F0F | MASK) - repeated_zp) ^ MASK;
}

// ── 2) LiquidQuant: la escala entra en el mismo IMAD ─────────────────────────────────────────
// Q_i8 = (Q_u4 * s_u8 + a) ^ 0x80808080, con a = 2^7 + min ya empaquetado por byte.
// Es valido porque al cuantizar en espacio unsigned todos los intermedios quedan en UINT8.
__device__ __forceinline__ void dequant_liquid(int q, int s_packed, int a_packed, int* out) {
  constexpr int MASK = 0x80808080;
  out[0] = ((q & 0x0F0F0F0F) * s_packed + a_packed) ^ MASK;
  q >>= 4;
  out[1] = ((q & 0x0F0F0F0F) * s_packed + a_packed) ^ MASK;
}

// ── 3) Marlin + la escala aplicada aparte, que es el costo REAL comparable ───────────────────
// Marlin escala el acumulador despues del mma, no cada peso; para comparar manzanas con manzanas
// se le suma aca una multiplicacion por la escala.
__device__ __forceinline__ void dequant_marlin_mas_escala(int q, int s, int* out) {
  constexpr int repeated_zp = 0x08080808;
  constexpr int MASK = 0x80808080;
  out[0] = (((q & 0x0F0F0F0F | MASK) - repeated_zp) ^ MASK) * s;
  q >>= 4;
  out[1] = (((q & 0x0F0F0F0F | MASK) - repeated_zp) ^ MASK) * s;
}

// ── 4) QServe: el nibble va CRUDO al tensor core, el offset se corrige en el acumulador ─────
// w = (q - 8) * s  =>  suma_k a_k*w_k = s * [ suma_k a_k*q_k  -  8 * suma_k a_k ]
// El mma hace la primera suma con q en [0,15] (positivos, validos como int8) y el "-8*suma(a)"
// se aplica UNA vez por acumulador, no por elemento. Es algebra exacta: cero perdida, a
// diferencia de LiquidQuant que paga 3,85% de error por meter la escala en el IMAD.
__device__ __forceinline__ void dequant_qserve(int q, int* out) {
  out[0] = q & 0x0F0F0F0F;
  out[1] = (q >> 4) & 0x0F0F0F0F;
}

// ── 5) QServe COMPLETO: desempaque crudo + la correccion del offset en el acumulador ────────
// Lo de arriba mide solo el desempaque. Aca se suma lo que cuesta la correccion, que es lo que
// decide si la ganancia neta existe: por cada acumulador hay que restar 8 * suma_a * escala.
// suma_a viene precalculada (un valor por fila y por grupo de K), asi que en el lazo es una
// multiplicacion y una resta sobre los 4 acumuladores del fragmento.
// OJO con la FRECUENCIA: la correccion NO va por cada desempaque. Es por (fila, grupo de K), y
// Marlin ya recorre los grupos aparte — con thread_k_blocks=8 y group_blocks=8 hay UN grupo por
// tile, o sea una correccion cada 32 mma. Medirla por cada 8 elementos la hace parecer 6x mas
// cara de lo que es: da 1,28x en vez de ~1,9x.
#define CADA 32

__device__ __forceinline__ void dequant_qserve_crudo(int q, int* out) {
  out[0] = q & 0x0F0F0F0F;
  out[1] = (q >> 4) & 0x0F0F0F0F;
}

__device__ __forceinline__ void correccion_offset(int suma_a, int esc, int* acc) {
  int corr = 8 * suma_a * esc;
#pragma unroll
  for (int j = 0; j < 4; j++) acc[j] -= corr;
}

template <int MODO>
__global__ void banco(const int* __restrict__ entrada, int* __restrict__ salida, int n) {
  int acc0 = 0, acc1 = 0;
  int accq[4] = {0, 0, 0, 0};   // acumulador del mma, para el modo QServe completo
  int base = blockIdx.x * blockDim.x + threadIdx.x;
  int q = entrada[base % n];
  int s = 0x01010101 * ((base & 7) + 1);
  int a = 0x80808080;
#pragma unroll 8
  for (int i = 0; i < REP; i++) {
    int out[2];
    if constexpr (MODO == 0) dequant_marlin(q ^ i, out);
    if constexpr (MODO == 1) dequant_liquid(q ^ i, s, a, out);
    if constexpr (MODO == 2) dequant_marlin_mas_escala(q ^ i, s, out);
    if constexpr (MODO == 3) dequant_qserve(q ^ i, out);
    // El acumulador es PERSISTENTE entre vueltas y se usa al final, asi que el compilador no
    // puede tirar la correccion. En la primera version acc[0]^acc[2] daba siempre 0 y nvcc la
    // borraba entera: t4 salia identico a t3, que fue la pista.
    if constexpr (MODO == 4) {
      dequant_qserve_crudo(q ^ i, out);
      if ((i % CADA) == 0) correccion_offset(s + i, (i & 255) + 1, accq);
    }
    acc0 += out[0];
    acc1 += out[1];
  }
  if constexpr (MODO == 4)
    salida[base] = accq[0] ^ accq[1] ^ accq[2] ^ accq[3] ^ acc0 ^ acc1;
  else
    salida[base] = acc0 ^ acc1;
}

template <int MODO>
float medir(const int* d_in, int* d_out, int n, int bloques, int hilos) {
  for (int i = 0; i < 5; i++) banco<MODO><<<bloques, hilos>>>(d_in, d_out, n);
  cudaDeviceSynchronize();
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  cudaEventRecord(a);
  for (int i = 0; i < 50; i++) banco<MODO><<<bloques, hilos>>>(d_in, d_out, n);
  cudaEventRecord(b);
  cudaEventSynchronize(b);
  float ms;
  cudaEventElapsedTime(&ms, a, b);
  return ms / 50.0f;
}

int main() {
  const int n = 4096, bloques = 82, hilos = 256;
  int *d_in, *d_out;
  cudaMalloc(&d_in, n * sizeof(int));
  cudaMalloc(&d_out, (size_t)bloques * hilos * sizeof(int));
  cudaMemset(d_in, 0x5A, n * sizeof(int));

  float t0 = medir<0>(d_in, d_out, n, bloques, hilos);
  float t1 = medir<1>(d_in, d_out, n, bloques, hilos);
  float t2 = medir<2>(d_in, d_out, n, bloques, hilos);
  float t3 = medir<3>(d_in, d_out, n, bloques, hilos);
  float t4 = medir<4>(d_in, d_out, n, bloques, hilos);

  double elem = (double)bloques * hilos * REP * 8;  // 8 elementos por vuelta
  printf("desempaque int4 -> int8, %d bloques x %d hilos x %d vueltas\n", bloques, hilos, REP);
  printf("  %-34s %8.3f ms   %6.2f Gelem/s   %s\n", "Marlin (sin escala)", t0,
         elem / (t0 * 1e6), "and/or/sub/xor");
  printf("  %-34s %8.3f ms   %6.2f Gelem/s   %s\n", "LiquidQuant (escala adentro)", t1,
         elem / (t1 * 1e6), "and/imad/xor");
  printf("  %-34s %8.3f ms   %6.2f Gelem/s   %s\n", "Marlin + escala aparte", t2,
         elem / (t2 * 1e6), "and/or/sub/xor/mul");
  printf("  %-34s %8.3f ms   %6.2f Gelem/s   %s\n", "QServe (nibble crudo, exacto)", t3,
         elem / (t3 * 1e6), "and/shr");
  printf("  %-34s %8.3f ms   %6.2f Gelem/s   %s\n", "QServe + correccion (lo real)", t4,
         elem / (t4 * 1e6), "and/shr + correccion 1 cada 32");
  printf("\n  LiquidQuant contra Marlin (sin escala): %.2fx   (cuesta 3,85%% de error)\n", t0 / t1);
  printf("  QServe solo desempaque:                %.2fx\n", t0 / t3);
  printf("  QServe con la correccion (NETO):       %.2fx   (exacto, sin perdida)\n", t2 / t4);
  cudaFree(d_in);
  cudaFree(d_out);
  return 0;
}
