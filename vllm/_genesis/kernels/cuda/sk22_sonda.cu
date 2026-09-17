// Sonda del layout de mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
//
// No se deduce de la doc: se MIDE. Cada hilo carga en su fragmento de B un valor unico que lo
// identifica (lane*8 + byte), y en A se pone un unico 1 en la posicion (fila, k) que se quiere
// sondear. Entonces
//
//     C[fila][n] = sum_k A[fila][k] * B[k][n] = B[k][n]
//
// y el valor que sale por C dice exactamente que (lane, byte) del fragmento estaba en B[k][n].
// Barriendo k de 0 a 31 y n de 0 a 7 sale el mapa completo.
//
// Lo mismo para A: se pone un 1 en el fragmento de un lane/byte concreto y se mira que columna
// de C se enciende.

#include <cstdint>

__device__ __forceinline__ void mma_s8(const uint32_t a[4], const uint32_t b[2], int32_t c[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}

// modo 0: sonda de B. A lleva un unico 1 en (fila_a, k_a); B lleva el identificador.
// modo 1: sonda de A. B lleva un unico 1 en su (lane_b, byte_b); A lleva el identificador.
extern "C" __global__ void sk22_sonda(int32_t* __restrict__ salida, int modo,
                                      int fila_a, int k_a, int lane_b, int byte_b) {
  const int lane = threadIdx.x;
  uint32_t a[4] = {0, 0, 0, 0};
  uint32_t b[2] = {0, 0};
  int32_t c[4] = {0, 0, 0, 0};

  if (modo == 0) {
    // B identificado: byte j del registro r del lane => valor 1 + lane*8 + r*4 + j
    for (int r = 0; r < 2; r++) {
      uint32_t v = 0;
      for (int j = 0; j < 4; j++) {
        int id = 1 + lane * 8 + r * 4 + j;
        v |= (uint32_t)(id & 0x7F) << (8 * j);   // 7 bits: entra en int8 sin signo
      }
      b[r] = v;
    }
    // A: un solo 1. El layout de A que se ASUME aca es el que la sonda de modo 1 confirma.
    // Para el barrido de B alcanza con encender el hilo/byte que corresponde a (fila_a, k_a)
    // segun el mapa que devuelve el modo 1, que se corre primero.
    int lane_obj = (fila_a % 8) * 4 + (k_a % 16) / 4;
    int reg_obj = (k_a / 16) * 2 + (fila_a / 8);
    int byte_obj = k_a % 4;
    if (lane == lane_obj) a[reg_obj] = (uint32_t)1 << (8 * byte_obj);
  } else {
    // A identificado, B con un solo 1
    for (int r = 0; r < 4; r++) {
      uint32_t v = 0;
      for (int j = 0; j < 4; j++) {
        int id = 1 + lane * 16 + r * 4 + j;
        v |= (uint32_t)(id & 0x7F) << (8 * j);
      }
      a[r] = v;
    }
    if (lane == lane_b) b[byte_b / 4] = (uint32_t)1 << (8 * (byte_b % 4));
  }

  mma_s8(a, b, c);

  // C de m16n8: el hilo `lane` tiene las filas lane/4 y lane/4+8, columnas (lane%4)*2 y +1
  for (int i = 0; i < 4; i++) salida[lane * 4 + i] = c[i];
}
