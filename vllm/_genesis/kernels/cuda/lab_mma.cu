// SPDX-License-Identifier: Apache-2.0
//
// LAB — mma de tensor cores en aislamiento: s8 m16n8k32 contra s4 m16n8k64.
//
// Cada bloque es UN warp (32 hilos) que ejecuta UNA multiplicacion de matrices
// completa del tamano del fragmento:
//     s8: A 16x32 (row-major, 4 regs/hilo)  B 32x8 (col-major, 2 regs/hilo)
//     s4: A 16x64 (row-major, 4 regs/hilo)  B 64x8 (col-major, 2 regs/hilo)
//     C 16x8 int32 (row-major, 4 regs/hilo) en los dos casos
// Los registros llegan ya empaquetados desde Python (la carga no se mide: se
// hace UNA vez antes del bucle). El bucle repite el mma REPS veces sobre los
// mismos registros, asi el tiempo es emision pura del pipe tensorial.
//
// Resultado por hilo: acc = REPS * C.

#include <cuda_pipeline.h>

extern "C" __global__ void __launch_bounds__(32)
lab_mma_s8(const unsigned* __restrict__ Ar, const unsigned* __restrict__ Br,
           int* __restrict__ out, int reps)
{
    const int tid = threadIdx.x;
    const size_t blk = blockIdx.x;
    const unsigned* a = Ar + blk * 32 * 4 + tid * 4;
    const unsigned* b = Br + blk * 32 * 2 + tid * 2;
    unsigned a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], b0 = b[0], b1 = b[1];
    int acc[4];
    acc[0] = 0; acc[1] = 0; acc[2] = 0; acc[3] = 0;
    for (int r = 0; r < reps; ++r) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    int* o = out + blk * 32 * 4 + tid * 4;
    o[0] = acc[0]; o[1] = acc[1]; o[2] = acc[2]; o[3] = acc[3];
}

extern "C" __global__ void __launch_bounds__(32)
lab_mma_s4(const unsigned* __restrict__ Ar, const unsigned* __restrict__ Br,
           int* __restrict__ out, int reps)
{
    const int tid = threadIdx.x;
    const size_t blk = blockIdx.x;
    const unsigned* a = Ar + blk * 32 * 4 + tid * 4;
    const unsigned* b = Br + blk * 32 * 2 + tid * 2;
    unsigned a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], b0 = b[0], b1 = b[1];
    int acc[4];
    acc[0] = 0; acc[1] = 0; acc[2] = 0; acc[3] = 0;
    for (int r = 0; r < reps; ++r) {
        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.s4.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    int* o = out + blk * 32 * 4 + tid * 4;
    o[0] = acc[0]; o[1] = acc[1]; o[2] = acc[2]; o[3] = acc[3];
}
