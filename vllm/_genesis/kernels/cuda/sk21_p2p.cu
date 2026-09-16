// SPDX-License-Identifier: Apache-2.0
//
// SK-21/p2p — copia entre placas escrita a mano, para poder elegir CUANTOS SM gasta.
//
// Por que hace falta:
//
//   * ``cudaMemcpyPeerAsync`` usa el MOTOR DE COPIA: no gasta SM y da 13,0 GB/s. Es la mejor
//     opcion cuando alcanza con una copia suelta.
//   * NCCL da 11,0 GB/s pero se toma muchos SM y por eso casi no solapa: medido, 25% contra un
//     GEMM, y 0% contra un bloque MLP que llena los 82 SM.
//   * Este kernel da 11,4 GB/s con OCHO bloques (10% de los SM) y solapa **93%**. Ese es el punto:
//     elegir cuanto SM gastar. Con 4 bloques baja a 9,6 GB/s y solapa 92%; de 16 para arriba el
//     ancho no sube mas y el solape empeora (64% con 16, 53% con 32).
//
// OJO CON LA MEMORIA: esto solo funciona sobre memoria compartida con ``cudaMalloc`` +
// ``cudaIpcGetMemHandle`` / ``cudaIpcOpenMemHandle``. Sobre un tensor de torch compartido con
// ``reduce_tensor`` el kernel da ACCESO ILEGAL y hasta el memcpyPeer baja a 5,6 GB/s.
// Ver tests/proto/sk21_ipc_crudo.py.
//
// Decisiones del kernel:
//  * TIRAR (leer remoto, escribir local) es el modo que se midio: 11,4 GB/s con 8 bloques.
//  * accesos de 16 bytes (``uint4``): es el tamano de transaccion que llena el enlace PCIe.
//  * ``cp.async`` NO sirve aca: es para global->shared de la MISMA placa.
//  * bucle con paso de grilla, asi el mismo kernel sirve para cualquier tamano con N fijo.
//
// La bandera se manda en un kernel aparte (``sk21_bandera``) DESPUES de la copia, en el mismo
// stream: asi el que recibe no la ve hasta que los datos llegaron. Lleva ``membar.sys`` para que
// la escritura de los datos sea visible en la otra placa antes que la bandera.
#ifndef BLOQUES
#define BLOQUES 8
#endif
#ifndef HILOS
#define HILOS 256
#endif

// copia de `n16` unidades de 16 bytes desde `org` (memoria de la OTRA placa) a `dst` (local)
extern "C" __global__ void __launch_bounds__(HILOS)
sk21_tirar(uint4* __restrict__ dst, const uint4* __restrict__ org, int n16) {
    // n16 va en int32: asi los pasa ptx_lab, y 2^31 unidades de 16 B son 32 GB de sobra
    const int paso = (int)(gridDim.x * HILOS);
    int i = (int)(blockIdx.x * HILOS + threadIdx.x);
    for (; i < n16; i += paso) {
        uint4 v;
        // ld.global a secas: el .nc (cache de solo lectura) NO vale sobre memoria de la otra
        // placa, da acceso ilegal.
        asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                     : "l"(org + i));
        asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};"
                     :: "l"(dst + i), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
    }
}

// Igual pero EMPUJANDO: lee local y escribe remoto. Se deja para poder medir las dos direcciones.
extern "C" __global__ void __launch_bounds__(HILOS)
sk21_empujar(uint4* __restrict__ dst, const uint4* __restrict__ org, int n16) {
    const int paso = (int)(gridDim.x * HILOS);
    int i = (int)(blockIdx.x * HILOS + threadIdx.x);
    for (; i < n16; i += paso) {
        uint4 v;
        asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                     : "l"(org + i));
        asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};"
                     :: "l"(dst + i), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
    }
}

// Pone `valor` en `bandera` (que vive en la OTRA placa) despues de que todo lo anterior sea
// visible en todo el sistema. Un solo hilo.
extern "C" __global__ void sk21_bandera(unsigned* bandera, unsigned valor) {
    asm volatile("membar.sys;" ::: "memory");
    asm volatile("st.global.u32 [%0], %1;" :: "l"(bandera), "r"(valor) : "memory");
}

// Diagnostico: la copia mas tonta posible, de a 4 bytes y sin PTX a mano. Sirve para separar
// "el kernel no puede tocar la memoria de la otra placa" de "el problema es la carga de 16 bytes".
extern "C" __global__ void sk21_simple(unsigned* dst, const unsigned* org, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) dst[i] = org[i];
}
