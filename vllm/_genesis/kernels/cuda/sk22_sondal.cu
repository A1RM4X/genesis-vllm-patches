// Sonda: ¿ldmatrix.x4.b16 entrega el fragmento A que quiere mma.m16n8k32.s8?
//
// La hipotesis es que si, tratando A int8 [16][32] como b16 [16][16]: el fragmento de A del mma
// entero, en unidades de 16 bits, tiene exactamente la convencion de las cuatro matrices 8x8 de
// ldmatrix.x4 (reg 0 = filas 0-7 cols 0-7, reg 1 = filas 8-15 cols 0-7, reg 2 y 3 idem con las
// columnas 8-15). Si vale, una instruccion reemplaza a las cuatro ld.shared de a 32 bits.
//
// No se razona: se compara contra la lectura escalar, que ya esta validada bit a bit.

#include <cstdint>

extern "C" __global__ void sk22_sondal(const int8_t* __restrict__ A, int32_t* __restrict__ sal,
                                       int swz) {
  __shared__ int8_t shA[16 * 32];
  const int lane = threadIdx.x;
  for (int i = lane; i < 16 * 32 / 4; i += 32) {
    int f = (i * 4) / 32, c = (i * 4) % 32;
    int cw = swz ? (c ^ ((f & 3) * 8)) : c;      // el swizzle XOR de CUTLASS, en bytes
    *reinterpret_cast<int32_t*>(&shA[f * 32 + cw]) =
        *reinterpret_cast<const int32_t*>(&A[f * 32 + c]);
  }
  __syncthreads();

  // (a) lectura escalar: el layout MEDIDO, cuatro ld.shared de 32 bits
  uint32_t esc[4];
  for (int r = 0; r < 4; r++) {
    int f = lane / 4 + (r % 2) * 8;
    int k = (lane % 4) * 4 + (r / 2) * 16;
    int kw = swz ? (k ^ ((f & 3) * 8)) : k;
    esc[r] = *reinterpret_cast<const uint32_t*>(&shA[f * 32 + kw]);
  }

  // (b) ldmatrix: una sola instruccion. El lane L apunta a la fila L%16, mitad L/16.
  int f = lane % 16, c = (lane / 16) * 16;
  int cw = swz ? (c ^ ((f & 3) * 8)) : c;
  uint32_t dir = static_cast<uint32_t>(__cvta_generic_to_shared(&shA[f * 32 + cw]));
  uint32_t ldm[4];
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(ldm[0]), "=r"(ldm[1]), "=r"(ldm[2]), "=r"(ldm[3])
               : "r"(dir));

  for (int r = 0; r < 4; r++) {
    sal[lane * 8 + r] = (int32_t)esc[r];
    sal[lane * 8 + 4 + r] = (int32_t)ldm[r];
  }
}
