# SPDX-License-Identifier: Apache-2.0
"""fused_quant_ptx sm_86 PTX 7.4 single-launch — sin fallback."""
from __future__ import annotations
import logging
import os
log = logging.getLogger("genesis.kernels.fused_quant_ptx")
_TARGET_SM = (8, 6)
_TARGET_SM_STR = "sm_86"
_PTX_VERSION = "7.4"
_PTX_ISA = "7.4"
_CUDA_ARCH_FLAG = "compute_86"
_CUDA_CODE_FLAG = "sm_86"
if "/usr/local/cuda/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ.setdefault("CUDA_PATH", "/usr/local/cuda")
try:
    import torch  # type: ignore
    _TORCH_OK = True
except Exception:
    torch = None  # type: ignore
    _TORCH_OK = False
_PTX_KERNEL_DOC = r"""
.version 7.4
.target sm_86
.address_size 64
// fused_quant_bf16_int8_ptx_kernel bf16 -> int8 per-token
// cvt.rn.f32.bf16 abs.f32 max.f32 shfl.sync rcp.approx.ftz.f32 mul.f32 cvt.rni.s32.f32 cvt.sat.s8.s32
// mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
"""
_CUTLASS_SKELETON = r"""
// cutlass int8 Gemm skeleton sm_86
// gemm mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
using GemmInt8 = cutlass::gemm::device::Gemm<int8_t,cutlass::layout::RowMajor,int8_t,cutlass::layout::ColumnMajor,int32_t,cutlass::layout::RowMajor,int32_t,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,cutlass::gemm::GemmShape<128,128,64>,cutlass::gemm::GemmShape<64,64,64>,cutlass::gemm::GemmShape<16,8,32>>;
"""
_CUDA_SRC = r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
__device__ __forceinline__ float bf16_to_f32_ptx(uint16_t h){float f; asm volatile("cvt.rn.f32.bf16 %0, %1;" : "=f"(f) : "h"(h)); return f;}
__device__ __forceinline__ float f32_abs_ptx(float x){float y; asm volatile("abs.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;}
__device__ __forceinline__ float f32_max_ptx(float a,float b){float c; asm volatile("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b)); return c;}
__device__ __forceinline__ float f32_rcp_ptx(float x){float y; asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;}
__device__ __forceinline__ float f32_mul_ptx(float a,float b){float c; asm volatile("mul.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b)); return c;}
__device__ __forceinline__ int f32_to_s32_rni_ptx(float x){int y; asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(y) : "f"(x)); return y;}
__device__ __forceinline__ int s32_to_s8_sat_ptx(int x){int y; asm volatile("cvt.sat.s8.s32 %0, %1;" : "=r"(y) : "r"(x)); return y;}
extern "C" __global__ void fused_quant_bf16_int8_ptx_kernel(const __nv_bfloat16* __restrict__ x,int8_t* __restrict__ y,float* __restrict__ scales,int M,int K){
    int row=blockIdx.x; if(row>=M) return;
    const uint16_t* row_x=reinterpret_cast<const uint16_t*>(x+(int64_t)row*K);
    int8_t* row_y=y+(int64_t)row*K;
    float tmax=0;
    for(int i=threadIdx.x;i<K;i+=blockDim.x){float f=bf16_to_f32_ptx(row_x[i]); float af=f32_abs_ptx(f); tmax=f32_max_ptx(tmax,af);}
    #pragma unroll
    for(int o=16;o>0;o>>=1){float other=__shfl_down_sync(0xffffffff,tmax,o); tmax=f32_max_ptx(tmax,other);}
    __shared__ float warp_max[32];
    int lane=threadIdx.x%32; int wid=threadIdx.x/32;
    if(lane==0) warp_max[wid]=tmax; __syncthreads();
    float bmax=0;
    if(wid==0){bmax=(lane< (blockDim.x+31)/32)? warp_max[lane]:0; #pragma unroll for(int o=16;o>0;o>>=1){float other=__shfl_down_sync(0xffffffff,bmax,o); bmax=f32_max_ptx(bmax,other);}}
    bmax=__shfl_sync(0xffffffff,bmax,0); __syncthreads();
    float amax=bmax;
    __shared__ float inv_s;
    if(threadIdx.x==0){float rcp=(amax==0)?0:f32_rcp_ptx(amax); float inv=f32_mul_ptx(rcp,127.0f); inv_s=inv; float s=amax/127.0f; if(s==0) s=1.0f; scales[row]=s;}
    __syncthreads(); float inv_scale=inv_s; inv_scale=__shfl_sync(0xffffffff,inv_scale,0);
    for(int i=threadIdx.x;i<K;i+=blockDim.x){float f=bf16_to_f32_ptx(row_x[i]); float scaled=f32_mul_ptx(f,inv_scale); int s32=f32_to_s32_rni_ptx(scaled); int s8=s32_to_s8_sat_ptx(s32); if(s8<-127) s8=-127; if(s8>127) s8=127; row_y[i]=(int8_t)s8;}
}
extern "C" void launch_fused_quant_ptx(int64_t x_ptr_int,int64_t y_ptr_int,int64_t scale_ptr_int,int M,int K,int dtype_code,int64_t stream_int){
    const void* x_ptr=reinterpret_cast<const void*>(x_ptr_int);
    void* y_ptr=reinterpret_cast<void*>(y_ptr_int);
    void* scale_ptr=reinterpret_cast<void*>(scale_ptr_int);
    cudaStream_t stream=reinterpret_cast<cudaStream_t>(stream_int);
    dim3 grid(M); dim3 block(256);
    fused_quant_bf16_int8_ptx_kernel<<<grid,block,0,stream>>>(reinterpret_cast<const __nv_bfloat16*>(x_ptr),reinterpret_cast<int8_t*>(y_ptr),reinterpret_cast<float*>(scale_ptr),M,K);
}
extern "C" const char* get_ptx_version_info(){return "PTX 7.4 sm_86 fused_quant_bf16_int8_ptx_kernel (cvt.rn.f32.bf16, abs, max+shfl, rcp, mul, cvt.rni.s32.f32, cvt.sat.s8.s32, mma.m16n8k32.s8 doc)";}
"""
_CPP_SRC = r"""
#include <cstdint>
extern "C" void launch_fused_quant_ptx(int64_t x_ptr,int64_t y_ptr,int64_t scale_ptr,int M,int K,int dtype_code,int64_t stream);
extern "C" const char* get_ptx_version_info();
"""
_CUDA_MODULE = None
_CUDA_LOAD_ERROR: str | None = None
_CUDA_LOAD_ATTEMPTED = False
def _try_load_cuda_module():
    global _CUDA_MODULE, _CUDA_LOAD_ERROR, _CUDA_LOAD_ATTEMPTED
    if _CUDA_LOAD_ATTEMPTED:
        return _CUDA_MODULE
    _CUDA_LOAD_ATTEMPTED = True
    if not _TORCH_OK or torch is None:
        _CUDA_LOAD_ERROR = "torch no disponible"
        return None
    try:
        import torch.utils.cpp_extension as cpp_ext  # type: ignore
    except Exception as e:
        _CUDA_LOAD_ERROR = f"cpp_extension no disponible: {e}"
        return None
    try:
        if not torch.cuda.is_available():
            _CUDA_LOAD_ERROR = "torch.cuda.is_available() == False"
            return None
    except Exception as e:
        _CUDA_LOAD_ERROR = f"cuda check fallo: {e}"
        return None
    try:
        mod = cpp_ext.load_inline(name="genesis_fused_quant_ptx_sm86", cpp_sources=_CPP_SRC, cuda_sources=_CUDA_SRC, functions=["launch_fused_quant_ptx","get_ptx_version_info"], extra_cflags=["-O3","-std=c++17"], extra_cuda_cflags=["-O3","-std=c++17","-gencode=arch=compute_86,code=sm_86","-gencode=arch=compute_86,code=compute_86","-lineinfo","--expt-relaxed-constexpr","-Xcompiler=-fPIC"], extra_ldflags=[], verbose=False)
        _CUDA_MODULE = mod
        log.info("fused_quant_ptx: modulo CUDA PTX sm_86 compilado OK (%s)", _PTX_VERSION)
        return mod
    except Exception as e:
        _CUDA_LOAD_ERROR = f"{type(e).__name__}: {e}"
        return None
def is_ptx_available() -> bool:
    return _try_load_cuda_module() is not None
def get_ptx_info() -> dict:
    return {"target_sm":_TARGET_SM_STR,"compute_capability":_TARGET_SM,"ptx_version":_PTX_VERSION,"ptx_isa":_PTX_ISA,"cuda_arch_flag":_CUDA_ARCH_FLAG,"cuda_code_flag":_CUDA_CODE_FLAG,"torch_ok":_TORCH_OK,"cuda_module_loaded":_CUDA_MODULE is not None,"cuda_load_error":_CUDA_LOAD_ERROR,"single_kernel":True,"fallback_triton":False,"triton_ok":False,"cupy_ok":False,"numba_ok":False,"ptx_ops":["cvt.rn.f32.bf16","abs.f32","max.f32","shfl.sync.bfly.b32 / shfl.sync.down.b32 (__shfl_down_sync)","rcp.approx.ftz.f32","mul.f32","cvt.rni.s32.f32","cvt.sat.s8.s32","mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 (doc, CUTLASS)"]}
def get_cuda_source() -> str:
    return _CUDA_SRC
def get_ptx_kernel_doc() -> str:
    return _PTX_KERNEL_DOC + "\n" + _CUTLASS_SKELETON
def quant_activation_per_token_ptx(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not _TORCH_OK or torch is None:
        raise RuntimeError("torch no disponible — no se puede cuantizar")
    if not isinstance(x, torch.Tensor):
        raise ValueError("quant_activation_per_token_ptx: x debe ser torch.Tensor")
    if x.dim() < 1:
        raise ValueError(f"quant_activation_per_token_ptx: x.dim >=1, got {x.dim()}")
    if x.numel() == 0:
        sshape = x.shape[:-1] + (1,) if x.dim() >= 1 else (1,)
        return torch.empty_like(x, dtype=torch.int8), torch.ones(sshape, dtype=torch.float32, device=x.device)
    if not getattr(x, "is_cuda", False) or not torch.cuda.is_available():
        raise RuntimeError("CUDA required — fallback eliminado (sin fallback torch/CPU)")
    mod = _try_load_cuda_module()
    if mod is None:
        raise RuntimeError(f"PTX module no disponible (sin fallback): {_CUDA_LOAD_ERROR}")
    orig_shape = x.shape
    k = int(orig_shape[-1])
    m = x.numel() // k if k else 0
    x_2d = x.reshape(m, k).contiguous()
    dtype_code = 0
    x_launch = x_2d
    if x_2d.dtype == torch.float16:
        try:
            x_launch = x_2d.to(torch.bfloat16).contiguous()
        except Exception:
            x_launch = x_2d
        dtype_code = 0
    elif x_2d.dtype == torch.float32:
        try:
            x_launch = x_2d.to(torch.bfloat16).contiguous()
        except Exception:
            x_launch = x_2d.to(torch.float16).contiguous()
        dtype_code = 0
    out_2d = torch.empty((m, k), dtype=torch.int8, device=x.device)
    scale_2d = torch.empty((m,), dtype=torch.float32, device=x.device)
    stream = torch.cuda.current_stream(x.device).cuda_stream if hasattr(torch.cuda.current_stream(x.device), "cuda_stream") else 0
    try:
        mod.launch_fused_quant_ptx(x_launch.data_ptr(), out_2d.data_ptr(), scale_2d.data_ptr(), int(m), int(k), int(dtype_code), int(stream) if isinstance(stream, int) else 0)
    except TypeError:
        mod.launch_fused_quant_ptx(x_launch.data_ptr(), out_2d.data_ptr(), scale_2d.data_ptr(), int(m), int(k), int(dtype_code), 0)
    return out_2d.reshape(orig_shape), scale_2d.reshape(orig_shape[:-1] + (1,)).contiguous()
__all__ = ["quant_activation_per_token_ptx","is_ptx_available","get_ptx_info","get_cuda_source","get_ptx_kernel_doc","_CUDA_SRC","_CPP_SRC","_PTX_KERNEL_DOC","_CUTLASS_SKELETON"]
