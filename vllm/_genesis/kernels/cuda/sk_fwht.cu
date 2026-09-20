// SPDX-License-Identifier: Apache-2.0
//
// sk_fwht.cu — Rotacion Hadamard ENTERA (Fast Walsh-Hadamard Transform) in-place para Q/K.
// Soporta D=128 (32 lanes x 4 dims) y D=256 (32 lanes x 8 dims), tanto fp16 como bf16.
// Sin punto flotante: todo el FWHT en int32 (Q14 / mariposas / desplazamiento de bits).

#include <cuda_fp16.h>
#include <cuda_bf16.h>

__device__ __forceinline__ unsigned bfly32(unsigned val, int lane_mask) {
    return __shfl_xor_sync(0xffffffff, val, lane_mask, 32);
}

__device__ __forceinline__ void fwht128_lanes(int* x, int lane) {
    int a0 = x[0], a1 = x[1];
    x[0] = a0 + a1; x[1] = a0 - a1;
    int a2 = x[2], a3 = x[3];
    x[2] = a2 + a3; x[3] = a2 - a3;
    
    a0 = x[0]; a2 = x[2];
    x[0] = a0 + a2; x[2] = a0 - a2;
    a1 = x[1]; a3 = x[3];
    x[1] = a1 + a3; x[3] = a1 - a3;

    #pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
        const int alto = (lane & m) != 0;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int t = (int)bfly32((unsigned)x[e], m);
            x[e] = alto ? (t - x[e]) : (x[e] + t);
        }
    }
}

__device__ __forceinline__ void fwht256_lanes(int* x, int lane) {
    #pragma unroll
    for (int len = 1; len < 8; len <<= 1) {
        for (int i = 0; i < 8; i += 2 * len) {
            for (int j = i; j < i + len; ++j) {
                const int a = x[j], b = x[j + len];
                x[j] = a + b; x[j + len] = a - b;
            }
        }
    }
    #pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
        const int alto = (lane & m) != 0;
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int t = (int)bfly32((unsigned)x[e], m);
            x[e] = alto ? (t - x[e]) : (x[e] + t);
        }
    }
}

// ──────────────────────────── FP16 Kernels ────────────────────────────

extern "C" __global__ void __launch_bounds__(128)
sk_fwht128_f16(
    unsigned short* __restrict__ data,      // [M, 128] fp16 bits in-place
    const int* __restrict__ signos,         // [128] +-1
    int M)
{
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 4 + warp_id;
    if (row >= M) return;

    unsigned short* ptr = data + (size_t)row * 128 + lane * 4;
    int x[4];

    #pragma unroll
    for (int e = 0; e < 4; ++e) {
        half h = *reinterpret_cast<const half*>(ptr + e);
        float f = __half2float(h);
        int xi = (int)roundf(f * 16384.0f);
        x[e] = xi * signos[lane * 4 + e];
    }

    fwht128_lanes(x, lane);

    #pragma unroll
    for (int e = 0; e < 4; ++e) {
        // Escala 1/sqrt(128) en punto fijo Q20 (round(2^20 / sqrt(128)) = 92682)
        long long prod = (long long)x[e] * 92682LL + 524288LL;
        int y = (int)(prod >> 20);
        float fo = (float)y * (1.0f / 16384.0f);
        *reinterpret_cast<half*>(ptr + e) = __float2half(fo);
    }
}

extern "C" __global__ void __launch_bounds__(128)
sk_fwht256_f16(
    unsigned short* __restrict__ data,      // [M, 256] fp16 bits in-place
    const int* __restrict__ signos,         // [256] +-1
    int M)
{
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 4 + warp_id;
    if (row >= M) return;

    unsigned short* ptr = data + (size_t)row * 256 + lane * 8;
    int x[8];

    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        half h = *reinterpret_cast<const half*>(ptr + e);
        float f = __half2float(h);
        int xi = (int)roundf(f * 16384.0f);
        x[e] = xi * signos[lane * 8 + e];
    }

    fwht256_lanes(x, lane);

    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        // Escala 1/sqrt(256) = 1/16: shift >> 4 exacto
        int y = (x[e] + 8) >> 4;
        float fo = (float)y * (1.0f / 16384.0f);
        *reinterpret_cast<half*>(ptr + e) = __float2half(fo);
    }
}

// ──────────────────────────── BF16 Kernels ────────────────────────────

extern "C" __global__ void __launch_bounds__(128)
sk_fwht128_bf16(
    unsigned short* __restrict__ data,      // [M, 128] bf16 bits in-place
    const int* __restrict__ signos,         // [128] +-1
    int M)
{
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 4 + warp_id;
    if (row >= M) return;

    unsigned short* ptr = data + (size_t)row * 128 + lane * 4;
    int x[4];

    #pragma unroll
    for (int e = 0; e < 4; ++e) {
        __nv_bfloat16 h = *reinterpret_cast<const __nv_bfloat16*>(ptr + e);
        float f = __bfloat162float(h);
        int xi = (int)roundf(f * 16384.0f);
        x[e] = xi * signos[lane * 4 + e];
    }

    fwht128_lanes(x, lane);

    #pragma unroll
    for (int e = 0; e < 4; ++e) {
        long long prod = (long long)x[e] * 92682LL + 524288LL;
        int y = (int)(prod >> 20);
        float fo = (float)y * (1.0f / 16384.0f);
        *reinterpret_cast<__nv_bfloat16*>(ptr + e) = __float2bfloat16(fo);
    }
}

extern "C" __global__ void __launch_bounds__(128)
sk_fwht256_bf16(
    unsigned short* __restrict__ data,      // [M, 256] bf16 bits in-place
    const int* __restrict__ signos,         // [256] +-1
    int M)
{
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 4 + warp_id;
    if (row >= M) return;

    unsigned short* ptr = data + (size_t)row * 256 + lane * 8;
    int x[8];

    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        __nv_bfloat16 h = *reinterpret_cast<const __nv_bfloat16*>(ptr + e);
        float f = __bfloat162float(h);
        int xi = (int)roundf(f * 16384.0f);
        x[e] = xi * signos[lane * 8 + e];
    }

    fwht256_lanes(x, lane);

    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        int y = (x[e] + 8) >> 4;
        float fo = (float)y * (1.0f / 16384.0f);
        *reinterpret_cast<__nv_bfloat16*>(ptr + e) = __float2bfloat16(fo);
    }
}
