// Sonda del layout de C en mma.m16n8k32: A[m][k] = m+1, B[k][n] = 1  =>  C[m][n] = (m+1)*32.
// El valor que sale por cada c[i] dice DIRECTAMENTE que fila m le toca a ese (lane, i).
#include <cstdint>
extern "C" __global__ void sk22_sondac(int32_t* __restrict__ salida) {
  const int lane = threadIdx.x;
  uint32_t a[4], b[2];
  int32_t c[4] = {0, 0, 0, 0};
  // A con el layout MEDIDO: fila = lane/4 + (reg%2)*8, k = (lane%4)*4 + byte + (reg/2)*16
  for (int r = 0; r < 4; r++) {
    int fila = lane / 4 + (r % 2) * 8;
    uint32_t v = 0;
    for (int j = 0; j < 4; j++) v |= (uint32_t)((fila + 1) & 0x7F) << (8 * j);
    a[r] = v;
  }
  b[0] = 0x01010101; b[1] = 0x01010101;
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
  for (int i = 0; i < 4; i++) salida[lane * 4 + i] = c[i];
}
