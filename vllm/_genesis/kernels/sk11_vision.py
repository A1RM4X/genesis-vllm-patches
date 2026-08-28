# SPDX-License-Identifier: Apache-2.0
"""SK-11 — VISION_BF16 — GEMM bf16 del ViT con epílogo fusionado, en GPU.

El ViT de Qwen3.5 está en ``modules_to_not_convert``: sus pesos quedan bf16 y
no hay Tensor Core INT8 que aprovechar. En bf16 puro, cuBLAS ya corre a ~90%
del pico de la 3090 (medido: 64 TFLOPS de 71 nominales), así que **un GEMM
Triton no le va a ganar** — la versión anterior de este archivo era un GEMM
naive 32x64x32 que corría 1.0-1.4x *más lento* que ``torch.matmul``.

Lo que sí se puede ganar es el epílogo. cuBLAS no fusiona bias+GELU, así que
cada proyección del ViT paga una pasada extra de lectura+escritura sobre M*N.
SK-11 ofrece las dos cosas y es explícito sobre cuál usar:

  * ``sk11_vision_gemm``      -> delega en cuBLAS. Es el óptimo para el GEMM
                                 desnudo y no hay motivo para reimplementarlo.
  * ``sk11_linear_gelu``      -> GEMM Triton con **bias + GELU tanh fusionados**
                                 en el epílogo: ahorra la pasada extra.
  * ``sk11_bias_gelu``        -> sólo el epílogo, para encadenar tras cuBLAS.

Diseño del GEMM bf16 (sm_86, mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32):
  * Acumulador fp32, tiles grandes, punteros que avanzan, grid 1-D con
    swizzle ``GROUP_M`` para reuso de B en L2.
  * shared = (BLOCK_M + BLOCK_N) * BLOCK_K * 2 bytes * num_stages.

Los kernels son branchless y no validan nada: se asume todo comprobado.
"""

from __future__ import annotations
import ctypes
import os
import subprocess
import tempfile
import threading

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Plomeria PTX. DUPLICADA A PROPOSITO en cada archivo de kernel: este modulo no
# importa plomeria compartida de ningun lado. Si hay que arreglar lo mismo en
# 18 archivos, se arregla en 18 archivos; es la decision explicita del
# proyecto -- codigo repetido antes que codigo compartido.
# ---------------------------------------------------------------------------

_ARCH = 86
_OPT = 3
# 48 KB es el maximo de shared DINAMICA por defecto en sm_86. Los tiles de
# prefill piden 73728 B (72 KB), asi que hay que pedirlo explicito con
# cuFuncSetAttribute o el launch falla con CUDA_ERROR_INVALID_VALUE (1).
_TECHO_SHARED = 49152
_ATRIB_MAX_SHARED = 8   # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES

_cerrojo = threading.Lock()
_libcuda_cache = None


def _libcuda():
    """libcuda cruda. El contexto lo inicializa torch en su primer op de GPU."""
    global _libcuda_cache
    if _libcuda_cache is None:
        _libcuda_cache = ctypes.CDLL("libcuda.so.1")
    return _libcuda_cache


def _ruta_ptxas() -> str:
    """El ptxas que trae Triton, no el del sistema.

    Triton emite PTX con una version de ISA concreta; el ptxas del CUDA
    instalado puede ser mas viejo o mas nuevo y rechazarlo.
    """
    base = os.path.dirname(__import__("triton").__file__)
    cand = os.path.join(base, "backends", "nvidia", "bin", "ptxas")
    return cand if os.path.exists(cand) else "ptxas"


def _ensamblar(ptx: str) -> bytes:
    """PTX -> cubin. Levanta RuntimeError con el stderr completo si falla."""
    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "k.ptx")
        fc = os.path.join(d, "k.cubin")
        with open(fp, "w") as f:
            f.write(ptx)
        r = subprocess.run([_ruta_ptxas(), "-arch=sm_%d" % _ARCH, "-O%d" % _OPT,
                            fp, "-o", fc], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("ptxas fallo:\n" + r.stderr)
        with open(fc, "rb") as f:
            return f.read()


class _Nativo:
    """Una variante de PTX embebida: ensambla bajo demanda y lanza por libcuda.

    :param caso: id de diagnostico.
    :param ptx: el asm completo.
    :param entry: nombre de la funcion dentro del modulo.
    :param warps: warps por bloque (blockDim.x = warps * 32).
    :param shared: bytes de shared dinamica que pide el cubin.
    :param abi: indices, sobre la lista completa de args del kernel original,
        de los params que SI viajan en el ABI, en orden. Salen de
        ``handle.src.constants``: son los que NO quedaron especializados.
    :param horneado: ``posicion -> valor`` de los params horneados en el cubin.
        Se verifican en cada lanzamiento: si llega otro valor el resultado
        seria incorrecto EN SILENCIO, asi que se levanta ValueError.
    :param div16: posiciones cuyo valor tiene que seguir siendo multiplo de 16.
        Triton lo asume al compilar (``tt.divisibility``) y lo usa para
        vectorizar los accesos. Con un valor que no cumple sale mal en
        silencio: el JIT recompilaria, el PTX embebido no puede.
    """

    def __init__(self, caso, ptx, entry, warps, shared, abi, horneado, div16):
        self.caso = caso
        self.ptx = ptx
        self.entry = entry
        self.warps = warps
        self.shared = shared
        self.abi = abi
        self.horneado = horneado
        self.div16 = div16
        self._fn = None
        self._mod = None

    def _cargar(self):
        """Camino rapido: si ya esta cargado no hace nada.

        El ensamblado vive aparte y marcado con ``torch.compiler.disable``
        porque usa un lock, y dynamo no sabe entrar a un context manager de
        `lock`: falla con "Unsupported context manager". El lanzamiento cae
        DENTRO de la region que vLLM compila (el forward del modelo), asi que
        con el `with` en el camino trazado el arranque se muere en
        ``profile_run`` -> ``_dummy_run``. Medido: ensamblar y cargar cuesta
        ~25 ms por variante y pasa una sola vez, asi que el corte de grafo del
        camino lento no se paga en regimen.
        """
        if self._fn is None:
            self._ensamblar_y_cargar()

    @torch.compiler.disable
    def _ensamblar_y_cargar(self):
        """Ensambla con ptxas y carga el modulo. Una sola vez, thread-safe."""
        with _cerrojo:
            if self._fn is not None:
                return
            cubin = _ensamblar(self.ptx)
            img = (ctypes.c_char * len(cubin)).from_buffer_copy(cubin)
            mod = ctypes.c_void_p()
            res = _libcuda().cuModuleLoadData(ctypes.byref(mod), img)
            if res != 0:
                raise RuntimeError("cuModuleLoadData(%s): %d" % (self.caso, res))
            fn = ctypes.c_void_p()
            res = _libcuda().cuModuleGetFunction(ctypes.byref(fn), mod,
                                                 self.entry.encode())
            if res != 0:
                raise RuntimeError("cuModuleGetFunction(%s): %d" % (self.caso, res))
            if self.shared > _TECHO_SHARED:
                res = _libcuda().cuFuncSetAttribute(
                    fn, _ATRIB_MAX_SHARED, ctypes.c_int(self.shared))
                if res != 0:
                    raise RuntimeError("cuFuncSetAttribute(%s, %dB): %d"
                                       % (self.caso, self.shared, res))
            self._mod, self._fn = mod, fn

    def __call__(self, grid, *args):
        """Lanza. ``args`` es la lista COMPLETA en orden de declaracion del
        kernel original; se consumen las posiciones de ``abi``."""
        self._cargar()
        for pos, val in self.horneado.items():
            if args[pos] != val:
                raise ValueError(
                    "%s: el cubin esta horneado con param[%d]=%r y llego %r; "
                    "este asm no aplica a estos inputs"
                    % (self.caso, pos, val, args[pos]))
        for pos in self.div16:
            if args[pos] % 16:
                raise ValueError(
                    "%s: param[%d]=%r tiene que ser multiplo de 16 (Triton lo "
                    "asumio al compilar para vectorizar)"
                    % (self.caso, pos, args[pos]))
        vals = []
        for i in self.abi:
            v = args[i]
            if isinstance(v, torch.Tensor):
                vals.append(ctypes.c_uint64(v.data_ptr()))
            elif isinstance(v, float):
                vals.append(ctypes.c_float(v))
            else:
                vals.append(ctypes.c_int32(int(v)))
        # ABI de Triton: params del kernel + global_scratch + profile_scratch.
        # global_scratch_size es 0 en estos kernels, asi que NULL es seguro.
        vals.append(ctypes.c_uint64(0))
        vals.append(ctypes.c_uint64(0))
        arr = (ctypes.c_void_p * len(vals))(
            *[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in vals])
        gx, gy, gz = (list(grid) + [1, 1])[:3]
        res = _libcuda().cuLaunchKernel(
            self._fn, gx, gy, gz, self.warps * 32, 1, 1,
            ctypes.c_uint(self.shared),
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
            arr, None)
        if res != 0:
            raise RuntimeError("cuLaunchKernel(%s): %d" % (self.caso, res))


def habilitado() -> bool:
    """Kill-switch: ``GENESIS_PTQ_NATIVO=0`` desactiva el camino PTX.

    Definido ACA, no importado: este modulo no comparte plomeria con nadie.
    """
    return os.environ.get("GENESIS_PTQ_NATIVO", "1").strip().lower() \
        not in ("0", "false", "no", "off")



SK_ID = "SK-11"
SK_NAME = "VISION_BF16"
VISION_HIDDEN: int = 1152

_CFG: tuple[tuple[int, int, int, int, int, int], ...] = (
    (16, 128, 64, 8, 4, 4),
    (64, 128, 64, 8, 8, 4),
    (128, 128, 64, 8, 8, 4),
    (128, 256, 64, 8, 8, 3),
)

BLOCK_M: int = 128
BLOCK_N: int = 128
BLOCK_K: int = 64

_ZERO: dict[torch.device, torch.Tensor] = {}


def _zero(device: torch.device) -> torch.Tensor:
    """Escalar cero por device, usado como bias neutro con stride 0."""
    z = _ZERO.get(device)
    if z is None:
        z = torch.zeros((), dtype=torch.bfloat16, device=device)
        _ZERO[device] = z
    return z


def _cfg(m: int) -> tuple[int, int, int, int, int, int]:
    return _CFG[(m > 32) + (m > 128) + (m > 512)]


@triton.jit
def _sk11_linear_gelu_kernel(
    a_ptr, b_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_out_m, stride_out_n, stride_bias,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    """GEMM bf16 con bias + GELU tanh fusionados en el epílogo."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_in_group = GROUP_M * num_pid_n
    first_m = (pid // pid_in_group) * GROUP_M
    group_m = min(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + ((pid % pid_in_group) % group_m)
    pid_n = (pid % pid_in_group) // group_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        acc += tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    x = acc + tl.load(bias_ptr + offs_n * stride_bias).to(tl.float32)[None, :]
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    y = 0.5 * x * (1.0 + (2.0 / (1.0 + tl.exp(-2.0 * inner)) - 1.0))

    out_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(
        out_ptr + out_m[:, None] * stride_out_m + out_n[None, :] * stride_out_n,
        y.to(out_ptr.dtype.element_ty),
        mask=(out_m[:, None] < M) & (out_n[None, :] < N),
    )


@triton.jit
def _sk11_bias_gelu_kernel(x_ptr, bias_ptr, out_ptr, N, stride_xm, stride_om, BLOCK: tl.constexpr):
    """bias + GELU tanh, una fila por programa."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    y = 0.5 * x * (1.0 + (2.0 / (1.0 + tl.exp(-2.0 * inner)) - 1.0))
    tl.store(out_ptr + row * stride_om + offs, y.to(tl.bfloat16), mask=mask)


def sk11_vision_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """GEMM bf16 desnudo. Delega en cuBLAS: es el óptimo en sm_86 para bf16."""
    return torch.matmul(a, b)


def sk11_linear_gelu(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """GEMM bf16 con bias + GELU tanh fusionados -> ``[M, N]`` bf16."""
    M, K = a.shape
    N = b.shape[1]
    bi = _zero(a.device) if bias is None else bias
    sb = 0 if bias is None else bi.stride(0)
    bm, bn, bk, gm, warps, stages = _cfg(M)
    out = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    _lanzar(grid, (bm, bn, bk, gm, warps, stages), False,
            a,
            b,
            bi,
            out,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            sb,
            bm,
            bn,
            bk,
            gm)
    return out


def sk11_bias_gelu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """bias + GELU tanh sobre ``[M, N]`` bf16, para encadenar tras cuBLAS."""
    M, N = x.shape
    out = torch.empty_like(x)
    _lanzar_quant0((M,),
            x, bias, out, N, x.stride(0), out.stride(0), triton.next_power_of_2(N))
    return out


passthrough_bf16 = sk11_vision_gemm
sk11_passthrough = sk11_vision_gemm
sk11_forward = sk11_vision_gemm

__all__ = [
    "SK_ID", "SK_NAME", "VISION_HIDDEN", "BLOCK_M", "BLOCK_N", "BLOCK_K",
    "sk11_vision_gemm", "sk11_linear_gelu", "sk11_bias_gelu",
    "passthrough_bf16", "sk11_passthrough", "sk11_forward",
]


# --- variantes PTX embebidas ---

_PTX_0 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk11_linear_gelu_kernel // -- Begin function _sk11_linear_gelu_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk11_linear_gelu_kernel
.visible .entry _sk11_linear_gelu_kernel(
	.param .u64 .ptr .global .align 1 _sk11_linear_gelu_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk11_linear_gelu_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk11_linear_gelu_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk11_linear_gelu_kernel_param_3,
	.param .u32 _sk11_linear_gelu_kernel_param_4,
	.param .u32 _sk11_linear_gelu_kernel_param_5,
	.param .u32 _sk11_linear_gelu_kernel_param_6,
	.param .u32 _sk11_linear_gelu_kernel_param_7,
	.param .u32 _sk11_linear_gelu_kernel_param_8,
	.param .u32 _sk11_linear_gelu_kernel_param_9,
	.param .u64 .ptr .global .align 1 _sk11_linear_gelu_kernel_param_10,
	.param .u64 .ptr .global .align 1 _sk11_linear_gelu_kernel_param_11
)
.reqntid 128
{
	.reg .pred 	%p<14>;
	.reg .b16 	%rs<26>;
	.reg .b32 	%r<505>;
	.reg .b64 	%rd<87>;
	.loc	1 66 0                          // sk11_vision.py:66:0
$L__func_begin0:
	.loc	1 66 0                          // sk11_vision.py:66:0

// %bb.0:
	ld.param.b32 	%r18, [_sk11_linear_gelu_kernel_param_9];
	ld.param.b32 	%r17, [_sk11_linear_gelu_kernel_param_5];
	ld.param.b32 	%r16, [_sk11_linear_gelu_kernel_param_4];
	ld.param.b64 	%rd14, [_sk11_linear_gelu_kernel_param_3];
	ld.param.b64 	%rd13, [_sk11_linear_gelu_kernel_param_2];
	ld.param.b64 	%rd12, [_sk11_linear_gelu_kernel_param_1];
	ld.param.b64 	%rd11, [_sk11_linear_gelu_kernel_param_0];
$L__tmp0:
	.loc	1 73 24                         // sk11_vision.py:73:24
	mov.u32 	%r49, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk11_vision.py:74:27 ]
	add.s32 	%r50, %r16, 15;
	.loc	2 43 30                         // standard.py:43:30 @[ sk11_vision.py:74:27 ]
	shr.s32 	%r51, %r50, 31;
	shr.u32 	%r52, %r51, 28;
	add.s32 	%r53, %r50, %r52;
	shr.s32 	%r54, %r53, 4;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk11_vision.py:75:27 ]
	add.s32 	%r55, %r17, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk11_vision.py:75:27 ]
	shr.s32 	%r56, %r55, 31;
	shr.u32 	%r57, %r56, 25;
	add.s32 	%r58, %r55, %r57;
	shr.s32 	%r59, %r58, 7;
$L__tmp3:
	.loc	1 76 29                         // sk11_vision.py:76:29
	shl.b32 	%r60, %r59, 3;
	ld.param.b32 	%r61, [_sk11_linear_gelu_kernel_param_6];
	ld.param.b32 	%r62, [_sk11_linear_gelu_kernel_param_7];
	.loc	1 77 22                         // sk11_vision.py:77:22
	div.s32 	%r63, %r49, %r60;
	ld.param.b32 	%r64, [_sk11_linear_gelu_kernel_param_8];
	.loc	1 77 38                         // sk11_vision.py:77:38
	shl.b32 	%r65, %r63, 3;
	.loc	1 78 30                         // sk11_vision.py:78:30
	sub.s32 	%r66, %r54, %r65;
	.loc	1 78 39                         // sk11_vision.py:78:39
	min.s32 	%r67, %r66, 8;
	.loc	1 79 30                         // sk11_vision.py:79:30
	mul.lo.s32 	%r68, %r63, %r60;
	sub.s32 	%r69, %r49, %r68;
	.loc	1 80 36                         // sk11_vision.py:80:36
	div.s32 	%r70, %r69, %r67;
	.loc	1 79 46                         // sk11_vision.py:79:46
	mul.lo.s32 	%r71, %r70, %r67;
	sub.s32 	%r72, %r69, %r71;
	.loc	1 79 23                         // sk11_vision.py:79:23
	add.s32 	%r73, %r72, %r65;
	.loc	1 82 22                         // sk11_vision.py:82:22
	shl.b32 	%r1, %r73, 4;
	.loc	1 82 45                         // sk11_vision.py:82:45
	mov.u32 	%r2, %tid.x;
	shr.u32 	%r74, %r2, 3;
	bfe.u32 	%r75, %r2, 3, 4;
	.loc	1 82 32                         // sk11_vision.py:82:32
	or.b32 	%r76, %r1, %r75;
	.loc	1 82 57                         // sk11_vision.py:82:57
	rem.s32 	%r77, %r76, %r16;
	.loc	1 83 22                         // sk11_vision.py:83:22
	shl.b32 	%r3, %r70, 7;
	.loc	1 83 45                         // sk11_vision.py:83:45
	and.b32 	%r4, %r2, 15;
	and.b32 	%r5, %r2, 127;
	.loc	1 83 32                         // sk11_vision.py:83:32
	or.b32 	%r78, %r3, %r75;
	or.b32 	%r79, %r78, 16;
	or.b32 	%r80, %r78, 32;
	or.b32 	%r81, %r78, 48;
	or.b32 	%r82, %r78, 64;
	or.b32 	%r83, %r78, 80;
	or.b32 	%r84, %r78, 96;
	or.b32 	%r85, %r74, %r3;
	or.b32 	%r86, %r85, 112;
	.loc	1 83 57                         // sk11_vision.py:83:57
	rem.s32 	%r87, %r78, %r17;
	rem.s32 	%r88, %r79, %r17;
	rem.s32 	%r89, %r80, %r17;
	rem.s32 	%r90, %r81, %r17;
	rem.s32 	%r91, %r82, %r17;
	rem.s32 	%r92, %r83, %r17;
	rem.s32 	%r93, %r84, %r17;
	rem.s32 	%r94, %r86, %r17;
	.loc	1 86 39                         // sk11_vision.py:86:39
	mul.lo.s32 	%r95, %r77, %r62;
	.loc	1 86 21                         // sk11_vision.py:86:21
	mad.wide.s32 	%rd42, %r95, 2, %rd11;
	.loc	1 86 58                         // sk11_vision.py:86:58
	and.b32 	%r6, %r2, 7;
	shl.b32 	%r96, %r6, 3;
	.loc	1 86 51                         // sk11_vision.py:86:51
	mul.wide.u32 	%rd43, %r96, 2;
	add.s64 	%rd15, %rd42, %rd43;
	.loc	1 87 21                         // sk11_vision.py:87:21
	add.s64 	%rd44, %rd12, %rd43;
	.loc	1 87 69                         // sk11_vision.py:87:69
	mul.lo.s32 	%r97, %r87, %r64;
	mul.lo.s32 	%r98, %r88, %r64;
	mul.lo.s32 	%r99, %r89, %r64;
	mul.lo.s32 	%r100, %r90, %r64;
	mul.lo.s32 	%r101, %r91, %r64;
	mul.lo.s32 	%r102, %r92, %r64;
	mul.lo.s32 	%r103, %r93, %r64;
	mul.lo.s32 	%r104, %r94, %r64;
	.loc	1 87 51                         // sk11_vision.py:87:51
	mad.wide.s32 	%rd16, %r97, 2, %rd44;
	mad.wide.s32 	%rd17, %r98, 2, %rd44;
	mad.wide.s32 	%rd18, %r99, 2, %rd44;
	mad.wide.s32 	%rd19, %r100, 2, %rd44;
	mad.wide.s32 	%rd20, %r101, 2, %rd44;
	mad.wide.s32 	%rd21, %r102, 2, %rd44;
	mad.wide.s32 	%rd22, %r103, 2, %rd44;
	mad.wide.s32 	%rd23, %r104, 2, %rd44;
$L__tmp4:
	.loc	2 43 17                         // standard.py:43:17 @[ sk11_vision.py:90:33 ]
	add.s32 	%r105, %r61, 63;
$L__tmp5:
	.loc	1 90 22                         // sk11_vision.py:90:22
	setp.lt.s32 	%p1, %r105, 64;
	setp.gt.s32 	%p2, %r105, 63;
	.loc	1 91 30                         // sk11_vision.py:91:30
	shl.b32 	%r109, %r5, 4;
	shl.b32 	%r8, %r2, 1;
	and.b32 	%r110, %r8, 112;
	xor.b32 	%r111, %r109, %r110;
	mov.b32 	%r112, global_smem;
	add.s32 	%r21, %r112, %r111;
	add.s32 	%r19, %r21, 49152;
	selp.b32 	%r20, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r19 + 0 ], [ %rd15 + 0 ], 0x10, %r20;
	// end inline asm
	cp.async.commit_group;
	.loc	1 91 47                         // sk11_vision.py:91:47
	// begin inline asm
	cp.async.cg.shared.global [ %r21 + 0 ], [ %rd16 + 0 ], 0x10, %r20;
	// end inline asm
	add.s32 	%r22, %r21, 2048;
	// begin inline asm
	cp.async.cg.shared.global [ %r22 + 0 ], [ %rd17 + 0 ], 0x10, %r20;
	// end inline asm
	add.s32 	%r23, %r21, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r23 + 0 ], [ %rd18 + 0 ], 0x10, %r20;
	// end inline asm
	add.s32 	%r24, %r21, 6144;
	// begin inline asm
	cp.async.cg.shared.global [ %r24 + 0 ], [ %rd19 + 0 ], 0x10, %r20;
	// end inline asm
	add.s32 	%r25, %r21, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r25 + 0 ], [ %rd20 + 0 ], 0x10, %r20;
	// end inline asm
	add.s32 	%r26, %r21, 10240;
	// begin inline asm
	cp.async.cg.shared.global [ %r26 + 0 ], [ %rd21 + 0 ], 0x10, %r20;
	// end inline asm
	add.s32 	%r27, %r21, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r27 + 0 ], [ %rd22 + 0 ], 0x10, %r20;
	// end inline asm
	add.s32 	%r28, %r21, 14336;
	// begin inline asm
	cp.async.cg.shared.global [ %r28 + 0 ], [ %rd23 + 0 ], 0x10, %r20;
	// end inline asm
	cp.async.commit_group;
	.loc	1 90 22                         // sk11_vision.py:90:22
	setp.gt.s32 	%p3, %r105, 127;
	.loc	1 92 18                         // sk11_vision.py:92:18
	add.s64 	%rd24, %rd15, 128;
	.loc	1 93 18                         // sk11_vision.py:93:18
	add.s64 	%rd25, %rd16, 128;
	add.s64 	%rd26, %rd17, 128;
	add.s64 	%rd27, %rd18, 128;
	add.s64 	%rd28, %rd19, 128;
	add.s64 	%rd29, %rd20, 128;
	add.s64 	%rd30, %rd21, 128;
	add.s64 	%rd31, %rd22, 128;
	add.s64 	%rd32, %rd23, 128;
	.loc	1 91 30                         // sk11_vision.py:91:30
	bar.sync 	0;
	add.s32 	%r29, %r21, 51200;
	selp.b32 	%r30, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r29 + 0 ], [ %rd24 + 0 ], 0x10, %r30;
	// end inline asm
	cp.async.commit_group;
	.loc	1 91 47                         // sk11_vision.py:91:47
	add.s32 	%r31, %r21, 16384;
	// begin inline asm
	cp.async.cg.shared.global [ %r31 + 0 ], [ %rd25 + 0 ], 0x10, %r30;
	// end inline asm
	add.s32 	%r32, %r21, 18432;
	// begin inline asm
	cp.async.cg.shared.global [ %r32 + 0 ], [ %rd26 + 0 ], 0x10, %r30;
	// end inline asm
	add.s32 	%r33, %r21, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r33 + 0 ], [ %rd27 + 0 ], 0x10, %r30;
	// end inline asm
	add.s32 	%r34, %r21, 22528;
	// begin inline asm
	cp.async.cg.shared.global [ %r34 + 0 ], [ %rd28 + 0 ], 0x10, %r30;
	// end inline asm
	add.s32 	%r35, %r21, 24576;
	// begin inline asm
	cp.async.cg.shared.global [ %r35 + 0 ], [ %rd29 + 0 ], 0x10, %r30;
	// end inline asm
	add.s32 	%r36, %r21, 26624;
	// begin inline asm
	cp.async.cg.shared.global [ %r36 + 0 ], [ %rd30 + 0 ], 0x10, %r30;
	// end inline asm
	add.s32 	%r37, %r21, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r37 + 0 ], [ %rd31 + 0 ], 0x10, %r30;
	// end inline asm
	add.s32 	%r38, %r21, 30720;
	// begin inline asm
	cp.async.cg.shared.global [ %r38 + 0 ], [ %rd32 + 0 ], 0x10, %r30;
	// end inline asm
	cp.async.commit_group;
	.loc	1 90 22                         // sk11_vision.py:90:22
	setp.gt.s32 	%p4, %r105, 191;
	.loc	1 92 18                         // sk11_vision.py:92:18
	add.s64 	%rd33, %rd15, 256;
	.loc	1 93 18                         // sk11_vision.py:93:18
	add.s64 	%rd34, %rd16, 256;
	add.s64 	%rd35, %rd17, 256;
	add.s64 	%rd36, %rd18, 256;
	add.s64 	%rd37, %rd19, 256;
	add.s64 	%rd38, %rd20, 256;
	add.s64 	%rd39, %rd21, 256;
	add.s64 	%rd40, %rd22, 256;
	add.s64 	%rd41, %rd23, 256;
	.loc	1 91 30                         // sk11_vision.py:91:30
	bar.sync 	0;
	add.s32 	%r39, %r21, 53248;
	selp.b32 	%r40, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r39 + 0 ], [ %rd33 + 0 ], 0x10, %r40;
	// end inline asm
	cp.async.commit_group;
	.loc	1 91 47                         // sk11_vision.py:91:47
	add.s32 	%r41, %r21, 32768;
	// begin inline asm
	cp.async.cg.shared.global [ %r41 + 0 ], [ %rd34 + 0 ], 0x10, %r40;
	// end inline asm
	add.s32 	%r42, %r21, 34816;
	// begin inline asm
	cp.async.cg.shared.global [ %r42 + 0 ], [ %rd35 + 0 ], 0x10, %r40;
	// end inline asm
	add.s32 	%r43, %r21, 36864;
	// begin inline asm
	cp.async.cg.shared.global [ %r43 + 0 ], [ %rd36 + 0 ], 0x10, %r40;
	// end inline asm
	add.s32 	%r44, %r21, 38912;
	// begin inline asm
	cp.async.cg.shared.global [ %r44 + 0 ], [ %rd37 + 0 ], 0x10, %r40;
	// end inline asm
	add.s32 	%r45, %r21, 40960;
	// begin inline asm
	cp.async.cg.shared.global [ %r45 + 0 ], [ %rd38 + 0 ], 0x10, %r40;
	// end inline asm
	add.s32 	%r46, %r21, 43008;
	// begin inline asm
	cp.async.cg.shared.global [ %r46 + 0 ], [ %rd39 + 0 ], 0x10, %r40;
	// end inline asm
	add.s32 	%r47, %r21, 45056;
	// begin inline asm
	cp.async.cg.shared.global [ %r47 + 0 ], [ %rd40 + 0 ], 0x10, %r40;
	// end inline asm
	add.s32 	%r48, %r21, 47104;
	// begin inline asm
	cp.async.cg.shared.global [ %r48 + 0 ], [ %rd41 + 0 ], 0x10, %r40;
	// end inline asm
	cp.async.commit_group;
	mov.b32 	%r488, 0f00000000;
	mov.b32 	%r489, %r488;
	mov.b32 	%r490, %r488;
	mov.b32 	%r491, %r488;
	mov.b32 	%r492, %r488;
	mov.b32 	%r493, %r488;
	mov.b32 	%r494, %r488;
	mov.b32 	%r495, %r488;
	mov.b32 	%r496, %r488;
	mov.b32 	%r497, %r488;
	mov.b32 	%r498, %r488;
	mov.b32 	%r499, %r488;
	mov.b32 	%r500, %r488;
	mov.b32 	%r501, %r488;
	mov.b32 	%r502, %r488;
	mov.b32 	%r503, %r488;
	.loc	1 90 22                         // sk11_vision.py:90:22
	@%p1 bra 	$L__BB0_3;
// %bb.1:                               // %.lr.ph
	.loc	1 0 22                          // sk11_vision.py:0:22
	cvt.s64.s32 	%rd1, %r95;
	cvt.s64.s32 	%rd2, %r97;
	cvt.s64.s32 	%rd3, %r98;
	cvt.s64.s32 	%rd4, %r99;
	cvt.s64.s32 	%rd5, %r100;
	cvt.s64.s32 	%rd6, %r101;
	cvt.s64.s32 	%rd7, %r102;
	cvt.s64.s32 	%rd8, %r103;
	cvt.s64.s32 	%rd9, %r104;
	shr.s32 	%r106, %r105, 31;
	shr.u32 	%r107, %r106, 26;
	add.s32 	%r108, %r105, %r107;
	shr.s32 	%r7, %r108, 6;
	add.s32 	%r9, %r7, -3;
	shl.b32 	%r113, %r4, 7;
	shl.b32 	%r114, %r6, 4;
	and.b32 	%r115, %r2, 16;
	xor.b32 	%r116, %r114, %r115;
	or.b32 	%r10, %r116, %r113;
	xor.b32 	%r11, %r10, 32;
	xor.b32 	%r12, %r10, 64;
	xor.b32 	%r13, %r10, 96;
	shl.b32 	%r117, %r6, 7;
	shl.b32 	%r118, %r2, 5;
	and.b32 	%r119, %r118, 3072;
	and.b32 	%r120, %r8, 48;
	or.b32 	%r121, %r117, %r119;
	xor.b32 	%r122, %r114, %r120;
	or.b32 	%r14, %r121, %r122;
	xor.b32 	%r15, %r14, 64;
	.loc	1 90 22                         // sk11_vision.py:90:22
	mul.wide.u32 	%rd10, %r6, 16;
	shl.b64 	%rd45, %rd9, 1;
	add.s64 	%rd46, %rd45, %rd12;
	add.s64 	%rd86, %rd46, 384;
	shl.b64 	%rd47, %rd8, 1;
	add.s64 	%rd48, %rd47, %rd12;
	add.s64 	%rd85, %rd48, 384;
	shl.b64 	%rd49, %rd7, 1;
	add.s64 	%rd50, %rd49, %rd12;
	add.s64 	%rd84, %rd50, 384;
	shl.b64 	%rd51, %rd6, 1;
	add.s64 	%rd52, %rd51, %rd12;
	add.s64 	%rd83, %rd52, 384;
	shl.b64 	%rd53, %rd5, 1;
	add.s64 	%rd54, %rd53, %rd12;
	add.s64 	%rd82, %rd54, 384;
	shl.b64 	%rd55, %rd4, 1;
	add.s64 	%rd56, %rd55, %rd12;
	add.s64 	%rd81, %rd56, 384;
	shl.b64 	%rd57, %rd3, 1;
	add.s64 	%rd58, %rd57, %rd12;
	add.s64 	%rd80, %rd58, 384;
	shl.b64 	%rd59, %rd2, 1;
	add.s64 	%rd60, %rd59, %rd12;
	add.s64 	%rd79, %rd60, 384;
	shl.b64 	%rd61, %rd1, 1;
	add.s64 	%rd62, %rd61, %rd11;
	add.s64 	%rd78, %rd62, 384;
	mov.b32 	%r504, 0;
	mov.b32 	%r488, 0f00000000;
	mov.b32 	%r487, 2;
	mov.b32 	%r486, -1;
	mov.b32 	%r489, %r488;
	mov.b32 	%r490, %r488;
	mov.b32 	%r491, %r488;
	mov.b32 	%r492, %r488;
	mov.b32 	%r493, %r488;
	mov.b32 	%r494, %r488;
	mov.b32 	%r495, %r488;
	mov.b32 	%r496, %r488;
	mov.b32 	%r497, %r488;
	mov.b32 	%r498, %r488;
	mov.b32 	%r499, %r488;
	mov.b32 	%r500, %r488;
	mov.b32 	%r501, %r488;
	mov.b32 	%r502, %r488;
	mov.b32 	%r503, %r488;
$L__BB0_2:                              // =>This Inner Loop Header: Depth=1
	setp.lt.s32 	%p5, %r504, %r9;
	add.s32 	%r181, %r486, 1;
	setp.gt.s32 	%p6, %r181, 2;
	selp.b32 	%r486, 0, %r181, %p6;
	.loc	1 91 30                         // sk11_vision.py:91:30
	cp.async.wait_group 	4;
	bar.sync 	0;
	shl.b32 	%r182, %r486, 11;
	add.s32 	%r183, %r112, %r182;
	add.s32 	%r184, %r183, %r10;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r123, %r124, %r125, %r126}, [%r184+49152];
	add.s32 	%r185, %r183, %r11;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r135, %r136, %r137, %r138}, [%r185+49152];
	add.s32 	%r186, %r183, %r12;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r147, %r148, %r149, %r150}, [%r186+49152];
	add.s32 	%r187, %r183, %r13;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r159, %r160, %r161, %r162}, [%r187+49152];
	.loc	1 91 47                         // sk11_vision.py:91:47
	shl.b32 	%r188, %r486, 14;
	add.s32 	%r189, %r112, %r188;
	add.s32 	%r190, %r189, %r14;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r127, %r128, %r139, %r140}, [%r190];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r129, %r130, %r141, %r142}, [%r190+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r131, %r132, %r143, %r144}, [%r190+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r133, %r134, %r145, %r146}, [%r190+12288];
	add.s32 	%r191, %r189, %r15;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r151, %r152, %r163, %r164}, [%r191];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r153, %r154, %r165, %r166}, [%r191+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r155, %r156, %r167, %r168}, [%r191+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r157, %r158, %r169, %r170}, [%r191+12288];
	.loc	1 91 39                         // sk11_vision.py:91:39
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r488, %r489, %r490, %r491 }, { %r123, %r124, %r125, %r126 }, { %r127, %r128 }, { %r488, %r489, %r490, %r491 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r492, %r493, %r494, %r495 }, { %r123, %r124, %r125, %r126 }, { %r129, %r130 }, { %r492, %r493, %r494, %r495 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r496, %r497, %r498, %r499 }, { %r123, %r124, %r125, %r126 }, { %r131, %r132 }, { %r496, %r497, %r498, %r499 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r500, %r501, %r502, %r503 }, { %r123, %r124, %r125, %r126 }, { %r133, %r134 }, { %r500, %r501, %r502, %r503 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r488, %r489, %r490, %r491 }, { %r135, %r136, %r137, %r138 }, { %r139, %r140 }, { %r488, %r489, %r490, %r491 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r492, %r493, %r494, %r495 }, { %r135, %r136, %r137, %r138 }, { %r141, %r142 }, { %r492, %r493, %r494, %r495 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r496, %r497, %r498, %r499 }, { %r135, %r136, %r137, %r138 }, { %r143, %r144 }, { %r496, %r497, %r498, %r499 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r500, %r501, %r502, %r503 }, { %r135, %r136, %r137, %r138 }, { %r145, %r146 }, { %r500, %r501, %r502, %r503 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r488, %r489, %r490, %r491 }, { %r147, %r148, %r149, %r150 }, { %r151, %r152 }, { %r488, %r489, %r490, %r491 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r492, %r493, %r494, %r495 }, { %r147, %r148, %r149, %r150 }, { %r153, %r154 }, { %r492, %r493, %r494, %r495 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r496, %r497, %r498, %r499 }, { %r147, %r148, %r149, %r150 }, { %r155, %r156 }, { %r496, %r497, %r498, %r499 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r500, %r501, %r502, %r503 }, { %r147, %r148, %r149, %r150 }, { %r157, %r158 }, { %r500, %r501, %r502, %r503 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r488, %r489, %r490, %r491 }, { %r159, %r160, %r161, %r162 }, { %r163, %r164 }, { %r488, %r489, %r490, %r491 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r492, %r493, %r494, %r495 }, { %r159, %r160, %r161, %r162 }, { %r165, %r166 }, { %r492, %r493, %r494, %r495 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r496, %r497, %r498, %r499 }, { %r159, %r160, %r161, %r162 }, { %r167, %r168 }, { %r496, %r497, %r498, %r499 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r500, %r501, %r502, %r503 }, { %r159, %r160, %r161, %r162 }, { %r169, %r170 }, { %r500, %r501, %r502, %r503 };
	// end inline asm
	.loc	1 93 18                         // sk11_vision.py:93:18
	add.s64 	%rd63, %rd78, %rd10;
	add.s64 	%rd64, %rd79, %rd10;
	add.s64 	%rd65, %rd80, %rd10;
	add.s64 	%rd66, %rd81, %rd10;
	add.s64 	%rd67, %rd82, %rd10;
	add.s64 	%rd68, %rd83, %rd10;
	add.s64 	%rd69, %rd84, %rd10;
	add.s64 	%rd70, %rd85, %rd10;
	.loc	1 90 22                         // sk11_vision.py:90:22
	add.s64 	%rd71, %rd86, %rd10;
	add.s32 	%r192, %r487, 1;
	setp.gt.s32 	%p7, %r192, 2;
	selp.b32 	%r487, 0, %r192, %p7;
	.loc	1 91 30                         // sk11_vision.py:91:30
	shl.b32 	%r193, %r487, 11;
	bar.sync 	0;
	add.s32 	%r194, %r21, %r193;
	add.s32 	%r171, %r194, 49152;
	selp.b32 	%r172, 16, 0, %p5;
	// begin inline asm
	cp.async.cg.shared.global [ %r171 + 0 ], [ %rd63 + 0 ], 0x10, %r172;
	// end inline asm
	cp.async.commit_group;
	.loc	1 91 47                         // sk11_vision.py:91:47
	shl.b32 	%r195, %r487, 14;
	add.s32 	%r173, %r21, %r195;
	// begin inline asm
	cp.async.cg.shared.global [ %r173 + 0 ], [ %rd64 + 0 ], 0x10, %r172;
	// end inline asm
	add.s32 	%r174, %r173, 2048;
	// begin inline asm
	cp.async.cg.shared.global [ %r174 + 0 ], [ %rd65 + 0 ], 0x10, %r172;
	// end inline asm
	add.s32 	%r175, %r173, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r175 + 0 ], [ %rd66 + 0 ], 0x10, %r172;
	// end inline asm
	add.s32 	%r176, %r173, 6144;
	// begin inline asm
	cp.async.cg.shared.global [ %r176 + 0 ], [ %rd67 + 0 ], 0x10, %r172;
	// end inline asm
	add.s32 	%r177, %r173, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r177 + 0 ], [ %rd68 + 0 ], 0x10, %r172;
	// end inline asm
	add.s32 	%r178, %r173, 10240;
	// begin inline asm
	cp.async.cg.shared.global [ %r178 + 0 ], [ %rd69 + 0 ], 0x10, %r172;
	// end inline asm
	add.s32 	%r179, %r173, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r179 + 0 ], [ %rd70 + 0 ], 0x10, %r172;
	// end inline asm
	add.s32 	%r180, %r173, 14336;
	// begin inline asm
	cp.async.cg.shared.global [ %r180 + 0 ], [ %rd71 + 0 ], 0x10, %r172;
	// end inline asm
	cp.async.commit_group;
	.loc	1 90 22                         // sk11_vision.py:90:22
	add.s32 	%r504, %r504, 1;
	add.s64 	%rd86, %rd86, 128;
	add.s64 	%rd85, %rd85, 128;
	add.s64 	%rd84, %rd84, 128;
	add.s64 	%rd83, %rd83, 128;
	add.s64 	%rd82, %rd82, 128;
	add.s64 	%rd81, %rd81, 128;
	add.s64 	%rd80, %rd80, 128;
	add.s64 	%rd79, %rd79, 128;
	add.s64 	%rd78, %rd78, 128;
	setp.ne.b32 	%p8, %r7, %r504;
	@%p8 bra 	$L__BB0_2;
$L__BB0_3:                              // %._crit_edge
	.loc	1 83 32                         // sk11_vision.py:83:32
	or.b32 	%r209, %r3, %r5;
	.loc	1 83 57                         // sk11_vision.py:83:57
	rem.s32 	%r210, %r209, %r17;
	.loc	1 83 45                         // sk11_vision.py:83:45
	shl.b32 	%r211, %r4, 3;
	.loc	1 83 32                         // sk11_vision.py:83:32
	or.b32 	%r212, %r3, %r211;
	.loc	1 82 45                         // sk11_vision.py:82:45
	shr.u32 	%r213, %r2, 4;
	bfe.u32 	%r214, %r2, 4, 3;
	.loc	1 82 32                         // sk11_vision.py:82:32
	or.b32 	%r215, %r214, %r1;
	or.b32 	%r216, %r215, 8;
	.loc	1 90 22                         // sk11_vision.py:90:22
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 95 33                         // sk11_vision.py:95:33
	mad.wide.s32 	%rd72, %r210, 2, %rd13;
	.loc	1 95 22                         // sk11_vision.py:95:22
	// begin inline asm
	mov.u16 %rs1, 0x0;
	ld.global.b16 { %rs1 }, [ %rd72 + 0 ];
	// end inline asm
	.loc	1 95 14                         // sk11_vision.py:95:14
	and.b32 	%r217, %r2, 31;
	shl.b32 	%r218, %r217, 2;
	and.b32 	%r219, %r213, 2;
	and.b32 	%r220, %r8, 128;
	add.s32 	%r221, %r112, %r218;
	add.s32 	%r222, %r221, %r219;
	add.s32 	%r196, %r222, %r220;
	// begin inline asm
	st.shared.b16 [ %r196 + 0 ], %rs1;
	// end inline asm
	bar.sync 	0;
	and.b32 	%r223, %r2, 3;
	shl.b32 	%r224, %r223, 3;
	and.b32 	%r225, %r2, 96;
	add.s32 	%r226, %r112, %r224;
	add.s32 	%r227, %r226, %r225;
	ld.shared.v4.b16 	{%rs18, %rs19, %rs20, %rs21}, [%r227];
	ld.shared.v4.b16 	{%rs22, %rs23, %rs24, %rs25}, [%r227+128];
	.loc	1 95 58                         // sk11_vision.py:95:58
	cvt.f32.bf16 	%r228, %rs18;
	cvt.f32.bf16 	%r229, %rs20;
	cvt.f32.bf16 	%r230, %rs19;
	cvt.f32.bf16 	%r231, %rs21;
	cvt.f32.bf16 	%r232, %rs22;
	cvt.f32.bf16 	%r233, %rs24;
	cvt.f32.bf16 	%r234, %rs23;
	cvt.f32.bf16 	%r235, %rs25;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r236, %r488, %r228;
	add.f32 	%r237, %r489, %r229;
	add.f32 	%r238, %r490, %r228;
	add.f32 	%r239, %r491, %r229;
	add.f32 	%r240, %r492, %r230;
	add.f32 	%r241, %r493, %r231;
	add.f32 	%r242, %r494, %r230;
	add.f32 	%r243, %r495, %r231;
	add.f32 	%r244, %r496, %r232;
	add.f32 	%r245, %r497, %r233;
	add.f32 	%r246, %r498, %r232;
	add.f32 	%r247, %r499, %r233;
	add.f32 	%r248, %r500, %r234;
	add.f32 	%r249, %r501, %r235;
	add.f32 	%r250, %r502, %r234;
	add.f32 	%r251, %r503, %r235;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r252, %r236, 0f3D372713;
	mul.f32 	%r253, %r237, 0f3D372713;
	mul.f32 	%r254, %r238, 0f3D372713;
	mul.f32 	%r255, %r239, 0f3D372713;
	mul.f32 	%r256, %r240, 0f3D372713;
	mul.f32 	%r257, %r241, 0f3D372713;
	mul.f32 	%r258, %r242, 0f3D372713;
	mul.f32 	%r259, %r243, 0f3D372713;
	mul.f32 	%r260, %r244, 0f3D372713;
	mul.f32 	%r261, %r245, 0f3D372713;
	mul.f32 	%r262, %r246, 0f3D372713;
	mul.f32 	%r263, %r247, 0f3D372713;
	mul.f32 	%r264, %r248, 0f3D372713;
	mul.f32 	%r265, %r249, 0f3D372713;
	mul.f32 	%r266, %r250, 0f3D372713;
	mul.f32 	%r267, %r251, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r268, %r236, %r252;
	mul.f32 	%r269, %r237, %r253;
	mul.f32 	%r270, %r238, %r254;
	mul.f32 	%r271, %r239, %r255;
	mul.f32 	%r272, %r240, %r256;
	mul.f32 	%r273, %r241, %r257;
	mul.f32 	%r274, %r242, %r258;
	mul.f32 	%r275, %r243, %r259;
	mul.f32 	%r276, %r244, %r260;
	mul.f32 	%r277, %r245, %r261;
	mul.f32 	%r278, %r246, %r262;
	mul.f32 	%r279, %r247, %r263;
	mul.f32 	%r280, %r248, %r264;
	mul.f32 	%r281, %r249, %r265;
	mul.f32 	%r282, %r250, %r266;
	mul.f32 	%r283, %r251, %r267;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r284, %r236, %r268, %r236;
	fma.rn.f32 	%r285, %r237, %r269, %r237;
	fma.rn.f32 	%r286, %r238, %r270, %r238;
	fma.rn.f32 	%r287, %r239, %r271, %r239;
	fma.rn.f32 	%r288, %r240, %r272, %r240;
	fma.rn.f32 	%r289, %r241, %r273, %r241;
	fma.rn.f32 	%r290, %r242, %r274, %r242;
	fma.rn.f32 	%r291, %r243, %r275, %r243;
	fma.rn.f32 	%r292, %r244, %r276, %r244;
	fma.rn.f32 	%r293, %r245, %r277, %r245;
	fma.rn.f32 	%r294, %r246, %r278, %r246;
	fma.rn.f32 	%r295, %r247, %r279, %r247;
	fma.rn.f32 	%r296, %r248, %r280, %r248;
	fma.rn.f32 	%r297, %r249, %r281, %r249;
	fma.rn.f32 	%r298, %r250, %r282, %r250;
	fma.rn.f32 	%r299, %r251, %r283, %r251;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r300, %r284, 0f3F4C422A;
	mul.f32 	%r301, %r285, 0f3F4C422A;
	mul.f32 	%r302, %r286, 0f3F4C422A;
	mul.f32 	%r303, %r287, 0f3F4C422A;
	mul.f32 	%r304, %r288, 0f3F4C422A;
	mul.f32 	%r305, %r289, 0f3F4C422A;
	mul.f32 	%r306, %r290, 0f3F4C422A;
	mul.f32 	%r307, %r291, 0f3F4C422A;
	mul.f32 	%r308, %r292, 0f3F4C422A;
	mul.f32 	%r309, %r293, 0f3F4C422A;
	mul.f32 	%r310, %r294, 0f3F4C422A;
	mul.f32 	%r311, %r295, 0f3F4C422A;
	mul.f32 	%r312, %r296, 0f3F4C422A;
	mul.f32 	%r313, %r297, 0f3F4C422A;
	mul.f32 	%r314, %r298, 0f3F4C422A;
	mul.f32 	%r315, %r299, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r316, %r236, 0f3F000000;
	mul.f32 	%r317, %r237, 0f3F000000;
	mul.f32 	%r318, %r238, 0f3F000000;
	mul.f32 	%r319, %r239, 0f3F000000;
	mul.f32 	%r320, %r240, 0f3F000000;
	mul.f32 	%r321, %r241, 0f3F000000;
	mul.f32 	%r322, %r242, 0f3F000000;
	mul.f32 	%r323, %r243, 0f3F000000;
	mul.f32 	%r324, %r244, 0f3F000000;
	mul.f32 	%r325, %r245, 0f3F000000;
	mul.f32 	%r326, %r246, 0f3F000000;
	mul.f32 	%r327, %r247, 0f3F000000;
	mul.f32 	%r328, %r248, 0f3F000000;
	mul.f32 	%r329, %r249, 0f3F000000;
	mul.f32 	%r330, %r250, 0f3F000000;
	mul.f32 	%r331, %r251, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r332, %r300, 0fC0000000;
	mul.f32 	%r333, %r301, 0fC0000000;
	mul.f32 	%r334, %r302, 0fC0000000;
	mul.f32 	%r335, %r303, 0fC0000000;
	mul.f32 	%r336, %r304, 0fC0000000;
	mul.f32 	%r337, %r305, 0fC0000000;
	mul.f32 	%r338, %r306, 0fC0000000;
	mul.f32 	%r339, %r307, 0fC0000000;
	mul.f32 	%r340, %r308, 0fC0000000;
	mul.f32 	%r341, %r309, 0fC0000000;
	mul.f32 	%r342, %r310, 0fC0000000;
	mul.f32 	%r343, %r311, 0fC0000000;
	mul.f32 	%r344, %r312, 0fC0000000;
	mul.f32 	%r345, %r313, 0fC0000000;
	mul.f32 	%r346, %r314, 0fC0000000;
	mul.f32 	%r347, %r315, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r348, %r332, 0f3FB8AA3B;
	ex2.approx.f32 	%r349, %r348;
	mul.f32 	%r350, %r333, 0f3FB8AA3B;
	ex2.approx.f32 	%r351, %r350;
	mul.f32 	%r352, %r334, 0f3FB8AA3B;
	ex2.approx.f32 	%r353, %r352;
	mul.f32 	%r354, %r335, 0f3FB8AA3B;
	ex2.approx.f32 	%r355, %r354;
	mul.f32 	%r356, %r336, 0f3FB8AA3B;
	ex2.approx.f32 	%r357, %r356;
	mul.f32 	%r358, %r337, 0f3FB8AA3B;
	ex2.approx.f32 	%r359, %r358;
	mul.f32 	%r360, %r338, 0f3FB8AA3B;
	ex2.approx.f32 	%r361, %r360;
	mul.f32 	%r362, %r339, 0f3FB8AA3B;
	ex2.approx.f32 	%r363, %r362;
	mul.f32 	%r364, %r340, 0f3FB8AA3B;
	ex2.approx.f32 	%r365, %r364;
	mul.f32 	%r366, %r341, 0f3FB8AA3B;
	ex2.approx.f32 	%r367, %r366;
	mul.f32 	%r368, %r342, 0f3FB8AA3B;
	ex2.approx.f32 	%r369, %r368;
	mul.f32 	%r370, %r343, 0f3FB8AA3B;
	ex2.approx.f32 	%r371, %r370;
	mul.f32 	%r372, %r344, 0f3FB8AA3B;
	ex2.approx.f32 	%r373, %r372;
	mul.f32 	%r374, %r345, 0f3FB8AA3B;
	ex2.approx.f32 	%r375, %r374;
	mul.f32 	%r376, %r346, 0f3FB8AA3B;
	ex2.approx.f32 	%r377, %r376;
	mul.f32 	%r378, %r347, 0f3FB8AA3B;
	ex2.approx.f32 	%r379, %r378;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r380, %r349, 0f3F800000;
	add.f32 	%r381, %r351, 0f3F800000;
	add.f32 	%r382, %r353, 0f3F800000;
	add.f32 	%r383, %r355, 0f3F800000;
	add.f32 	%r384, %r357, 0f3F800000;
	add.f32 	%r385, %r359, 0f3F800000;
	add.f32 	%r386, %r361, 0f3F800000;
	add.f32 	%r387, %r363, 0f3F800000;
	add.f32 	%r388, %r365, 0f3F800000;
	add.f32 	%r389, %r367, 0f3F800000;
	add.f32 	%r390, %r369, 0f3F800000;
	add.f32 	%r391, %r371, 0f3F800000;
	add.f32 	%r392, %r373, 0f3F800000;
	add.f32 	%r393, %r375, 0f3F800000;
	add.f32 	%r394, %r377, 0f3F800000;
	add.f32 	%r395, %r379, 0f3F800000;
	mov.b32 	%r396, 0f40000000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r397, %r396, %r380;
	div.full.f32 	%r398, %r396, %r381;
	div.full.f32 	%r399, %r396, %r382;
	div.full.f32 	%r400, %r396, %r383;
	div.full.f32 	%r401, %r396, %r384;
	div.full.f32 	%r402, %r396, %r385;
	div.full.f32 	%r403, %r396, %r386;
	div.full.f32 	%r404, %r396, %r387;
	div.full.f32 	%r405, %r396, %r388;
	div.full.f32 	%r406, %r396, %r389;
	div.full.f32 	%r407, %r396, %r390;
	div.full.f32 	%r408, %r396, %r391;
	div.full.f32 	%r409, %r396, %r392;
	div.full.f32 	%r410, %r396, %r393;
	div.full.f32 	%r411, %r396, %r394;
	div.full.f32 	%r412, %r396, %r395;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r413, %r397, 0fBF800000;
	add.f32 	%r414, %r398, 0fBF800000;
	add.f32 	%r415, %r399, 0fBF800000;
	add.f32 	%r416, %r400, 0fBF800000;
	add.f32 	%r417, %r401, 0fBF800000;
	add.f32 	%r418, %r402, 0fBF800000;
	add.f32 	%r419, %r403, 0fBF800000;
	add.f32 	%r420, %r404, 0fBF800000;
	add.f32 	%r421, %r405, 0fBF800000;
	add.f32 	%r422, %r406, 0fBF800000;
	add.f32 	%r423, %r407, 0fBF800000;
	add.f32 	%r424, %r408, 0fBF800000;
	add.f32 	%r425, %r409, 0fBF800000;
	add.f32 	%r426, %r410, 0fBF800000;
	add.f32 	%r427, %r411, 0fBF800000;
	add.f32 	%r428, %r412, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r429, %r413, 0f3F800000;
	add.f32 	%r430, %r414, 0f3F800000;
	add.f32 	%r431, %r415, 0f3F800000;
	add.f32 	%r432, %r416, 0f3F800000;
	add.f32 	%r433, %r417, 0f3F800000;
	add.f32 	%r434, %r418, 0f3F800000;
	add.f32 	%r435, %r419, 0f3F800000;
	add.f32 	%r436, %r420, 0f3F800000;
	add.f32 	%r437, %r421, 0f3F800000;
	add.f32 	%r438, %r422, 0f3F800000;
	add.f32 	%r439, %r423, 0f3F800000;
	add.f32 	%r440, %r424, 0f3F800000;
	add.f32 	%r441, %r425, 0f3F800000;
	add.f32 	%r442, %r426, 0f3F800000;
	add.f32 	%r443, %r427, 0f3F800000;
	add.f32 	%r444, %r428, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r445, %r316, %r429;
	mul.f32 	%r446, %r317, %r430;
	mul.f32 	%r447, %r318, %r431;
	mul.f32 	%r448, %r319, %r432;
	mul.f32 	%r449, %r320, %r433;
	mul.f32 	%r450, %r321, %r434;
	mul.f32 	%r451, %r322, %r435;
	mul.f32 	%r452, %r323, %r436;
	mul.f32 	%r453, %r324, %r437;
	mul.f32 	%r454, %r325, %r438;
	mul.f32 	%r455, %r326, %r439;
	mul.f32 	%r456, %r327, %r440;
	mul.f32 	%r457, %r328, %r441;
	mul.f32 	%r458, %r329, %r442;
	mul.f32 	%r459, %r330, %r443;
	mul.f32 	%r460, %r331, %r444;
	.loc	1 104 31                        // sk11_vision.py:104:31
	setp.lt.s32 	%p11, %r215, %r16;
	setp.lt.s32 	%p12, %r216, %r16;
	.loc	1 104 54                        // sk11_vision.py:104:54
	setp.lt.s32 	%p13, %r212, %r17;
	.loc	1 104 37                        // sk11_vision.py:104:37
	and.pred 	%p9, %p11, %p13;
	and.pred 	%p10, %p12, %p13;
	.loc	1 102 35                        // sk11_vision.py:102:35
	mul.lo.s32 	%r461, %r215, %r18;
	mul.lo.s32 	%r462, %r216, %r18;
	.loc	1 102 18                        // sk11_vision.py:102:18
	mad.wide.s32 	%rd75, %r461, 2, %rd14;
	mad.wide.s32 	%rd76, %r462, 2, %rd14;
	.loc	1 102 50                        // sk11_vision.py:102:50
	mul.wide.s32 	%rd77, %r212, 2;
	add.s64 	%rd73, %rd75, %rd77;
	add.s64 	%rd74, %rd76, %rd77;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16.f32 	%rs2, %r445;
	cvt.rn.bf16.f32 	%rs3, %r446;
	cvt.rn.bf16.f32 	%rs4, %r447;
	cvt.rn.bf16.f32 	%rs5, %r448;
	cvt.rn.bf16.f32 	%rs6, %r449;
	cvt.rn.bf16.f32 	%rs7, %r450;
	cvt.rn.bf16.f32 	%rs8, %r451;
	cvt.rn.bf16.f32 	%rs9, %r452;
	cvt.rn.bf16.f32 	%rs10, %r453;
	cvt.rn.bf16.f32 	%rs11, %r454;
	cvt.rn.bf16.f32 	%rs12, %r455;
	cvt.rn.bf16.f32 	%rs13, %r456;
	cvt.rn.bf16.f32 	%rs14, %r457;
	cvt.rn.bf16.f32 	%rs15, %r458;
	cvt.rn.bf16.f32 	%rs16, %r459;
	cvt.rn.bf16.f32 	%rs17, %r460;
	.loc	1 103 8                         // sk11_vision.py:103:8
	bar.sync 	0;
	shl.b32 	%r463, %r223, 10;
	shl.b32 	%r464, %r2, 6;
	and.b32 	%r465, %r464, 768;
	shl.b32 	%r466, %r217, 3;
	or.b32 	%r467, %r465, %r466;
	xor.b32 	%r468, %r467, %r225;
	or.b32 	%r469, %r468, %r463;
	add.s32 	%r197, %r112, %r469;
	// begin inline asm
	st.shared.v4.b16 [ %r197 + 0 ], { %rs2, %rs3, %rs4, %rs5 };
	// end inline asm
	xor.b32 	%r470, %r469, 8;
	add.s32 	%r198, %r112, %r470;
	// begin inline asm
	st.shared.v4.b16 [ %r198 + 0 ], { %rs6, %rs7, %rs8, %rs9 };
	// end inline asm
	xor.b32 	%r471, %r469, 16;
	add.s32 	%r199, %r112, %r471;
	// begin inline asm
	st.shared.v4.b16 [ %r199 + 0 ], { %rs10, %rs11, %rs12, %rs13 };
	// end inline asm
	xor.b32 	%r472, %r469, 24;
	add.s32 	%r200, %r112, %r472;
	// begin inline asm
	st.shared.v4.b16 [ %r200 + 0 ], { %rs14, %rs15, %rs16, %rs17 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r473, %r223, 5;
	shl.b32 	%r474, %r2, 4;
	and.b32 	%r475, %r474, 768;
	and.b32 	%r476, %r8, 248;
	or.b32 	%r477, %r473, %r475;
	xor.b32 	%r478, %r477, %r476;
	add.s32 	%r479, %r112, %r478;
	ld.shared.v2.b32 	{%r201, %r205}, [%r479];
	xor.b32 	%r480, %r478, 8;
	add.s32 	%r481, %r112, %r480;
	ld.shared.v2.b32 	{%r202, %r206}, [%r481+1024];
	xor.b32 	%r482, %r478, 16;
	add.s32 	%r483, %r112, %r482;
	ld.shared.v2.b32 	{%r203, %r207}, [%r483+2048];
	xor.b32 	%r484, %r478, 24;
	add.s32 	%r485, %r112, %r484;
	ld.shared.v2.b32 	{%r204, %r208}, [%r485+3072];
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd73 + 0 ], { %r201, %r202, %r203, %r204 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd74 + 0 ], { %r205, %r206, %r207, %r208 };
	// end inline asm
	.loc	1 101 4                         // sk11_vision.py:101:4
	ret;
$L__tmp6:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk11_vision.py"
	.file	2 "/usr/local/lib/python3.12/dist-packages/triton/language/standard.py"
	.section	.debug_abbrev
	{
.b8 1                                   // Abbreviation Code
.b8 17                                  // DW_TAG_compile_unit
.b8 1                                   // DW_CHILDREN_yes
.b8 37                                  // DW_AT_producer
.b8 8                                   // DW_FORM_string
.b8 19                                  // DW_AT_language
.b8 5                                   // DW_FORM_data2
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 16                                  // DW_AT_stmt_list
.b8 6                                   // DW_FORM_data4
.b8 27                                  // DW_AT_comp_dir
.b8 8                                   // DW_FORM_string
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 2                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 0                                   // DW_CHILDREN_no
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 32                                  // DW_AT_inline
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 3                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 1                                   // DW_CHILDREN_yes
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 4                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 0                                   // DW_CHILDREN_no
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 11                                  // DW_FORM_data1
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 186                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xb3 DW_TAG_compile_unit
.b8 116                                 // DW_AT_producer
.b8 114
.b8 105
.b8 116
.b8 111
.b8 110
.b8 0
.b8 2                                   // DW_AT_language
.b8 0
.b8 115                                 // DW_AT_name
.b8 107
.b8 49
.b8 49
.b8 95
.b8 118
.b8 105
.b8 115
.b8 105
.b8 111
.b8 110
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 114
.b8 101
.b8 112
.b8 111
.b8 47
.b8 118
.b8 108
.b8 108
.b8 109
.b8 47
.b8 95
.b8 103
.b8 101
.b8 110
.b8 101
.b8 115
.b8 105
.b8 115
.b8 47
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 115
.b8 0
.b8 2                                   // Abbrev [2] 0x44:0x1b DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 49
.b8 49
.b8 95
.b8 108
.b8 105
.b8 110
.b8 101
.b8 97
.b8 114
.b8 95
.b8 103
.b8 101
.b8 108
.b8 117
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x5f:0x5e DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 68                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x74:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 74                                  // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x8c:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 75                                  // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0xa4:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp4                           // DW_AT_low_pc
.b64 $L__tmp5                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 90                                  // DW_AT_call_line
.b8 33                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_0 = _Nativo(
    "sk11_vision/tile16x128x64_shift0_abi10",
    _PTX_0, "_sk11_linear_gelu_kernel",
    warps=4, shared=55296,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 10, 11],
    horneado={8: 1, 9: 1, 12: 1, 13: 1, 14: 16, 15: 128, 16: 64, 17: 8},
    div16=[5, 6, 7, 10, 11],
)

_PTX_1 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk11_linear_gelu_kernel // -- Begin function _sk11_linear_gelu_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk11_linear_gelu_kernel
.visible .entry _sk11_linear_gelu_kernel(
	.param .u64 .ptr .global .align 1 _sk11_linear_gelu_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk11_linear_gelu_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk11_linear_gelu_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk11_linear_gelu_kernel_param_3,
	.param .u32 _sk11_linear_gelu_kernel_param_4,
	.param .u32 _sk11_linear_gelu_kernel_param_5,
	.param .u32 _sk11_linear_gelu_kernel_param_6,
	.param .u32 _sk11_linear_gelu_kernel_param_7,
	.param .u32 _sk11_linear_gelu_kernel_param_8,
	.param .u32 _sk11_linear_gelu_kernel_param_9,
	.param .u64 .ptr .global .align 1 _sk11_linear_gelu_kernel_param_10,
	.param .u64 .ptr .global .align 1 _sk11_linear_gelu_kernel_param_11
)
.reqntid 256
{
	.reg .pred 	%p<18>;
	.reg .b16 	%rs<41>;
	.reg .b32 	%r<763>;
	.reg .b64 	%rd<71>;
	.loc	1 66 0                          // sk11_vision.py:66:0
$L__func_begin0:
	.loc	1 66 0                          // sk11_vision.py:66:0

// %bb.0:
	ld.param.b32 	%r19, [_sk11_linear_gelu_kernel_param_9];
	ld.param.b32 	%r18, [_sk11_linear_gelu_kernel_param_5];
	ld.param.b32 	%r17, [_sk11_linear_gelu_kernel_param_4];
	ld.param.b64 	%rd11, [_sk11_linear_gelu_kernel_param_3];
	ld.param.b64 	%rd10, [_sk11_linear_gelu_kernel_param_2];
	ld.param.b64 	%rd9, [_sk11_linear_gelu_kernel_param_1];
	ld.param.b64 	%rd8, [_sk11_linear_gelu_kernel_param_0];
$L__tmp0:
	.loc	1 73 24                         // sk11_vision.py:73:24
	mov.u32 	%r41, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk11_vision.py:74:27 ]
	add.s32 	%r42, %r17, 63;
	.loc	2 43 30                         // standard.py:43:30 @[ sk11_vision.py:74:27 ]
	shr.s32 	%r43, %r42, 31;
	shr.u32 	%r44, %r43, 26;
	add.s32 	%r45, %r42, %r44;
	shr.s32 	%r46, %r45, 6;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk11_vision.py:75:27 ]
	add.s32 	%r47, %r18, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk11_vision.py:75:27 ]
	shr.s32 	%r48, %r47, 31;
	shr.u32 	%r49, %r48, 25;
	add.s32 	%r50, %r47, %r49;
	shr.s32 	%r51, %r50, 7;
$L__tmp3:
	.loc	1 76 29                         // sk11_vision.py:76:29
	shl.b32 	%r52, %r51, 3;
	ld.param.b32 	%r53, [_sk11_linear_gelu_kernel_param_6];
	ld.param.b32 	%r54, [_sk11_linear_gelu_kernel_param_7];
	.loc	1 77 22                         // sk11_vision.py:77:22
	div.s32 	%r55, %r41, %r52;
	ld.param.b32 	%r56, [_sk11_linear_gelu_kernel_param_8];
	.loc	1 77 38                         // sk11_vision.py:77:38
	shl.b32 	%r57, %r55, 3;
	.loc	1 78 30                         // sk11_vision.py:78:30
	sub.s32 	%r58, %r46, %r57;
	.loc	1 78 39                         // sk11_vision.py:78:39
	min.s32 	%r59, %r58, 8;
	.loc	1 79 30                         // sk11_vision.py:79:30
	mul.lo.s32 	%r60, %r55, %r52;
	sub.s32 	%r61, %r41, %r60;
	.loc	1 80 36                         // sk11_vision.py:80:36
	div.s32 	%r62, %r61, %r59;
	.loc	1 79 46                         // sk11_vision.py:79:46
	mul.lo.s32 	%r63, %r62, %r59;
	sub.s32 	%r64, %r61, %r63;
	.loc	1 79 23                         // sk11_vision.py:79:23
	add.s32 	%r65, %r64, %r57;
	.loc	1 82 22                         // sk11_vision.py:82:22
	shl.b32 	%r1, %r65, 6;
	.loc	1 82 45                         // sk11_vision.py:82:45
	mov.u32 	%r2, %tid.x;
	shr.u32 	%r66, %r2, 3;
	bfe.u32 	%r67, %r2, 3, 5;
	or.b32 	%r68, %r67, 32;
	.loc	1 82 32                         // sk11_vision.py:82:32
	or.b32 	%r69, %r1, %r67;
	or.b32 	%r70, %r1, %r68;
	.loc	1 82 57                         // sk11_vision.py:82:57
	rem.s32 	%r71, %r69, %r17;
	rem.s32 	%r72, %r70, %r17;
	.loc	1 83 22                         // sk11_vision.py:83:22
	shl.b32 	%r3, %r62, 7;
	.loc	1 83 45                         // sk11_vision.py:83:45
	and.b32 	%r4, %r2, 96;
	and.b32 	%r5, %r2, 15;
	.loc	1 83 32                         // sk11_vision.py:83:32
	or.b32 	%r73, %r3, %r67;
	or.b32 	%r74, %r3, %r68;
	or.b32 	%r75, %r73, 64;
	or.b32 	%r76, %r3, %r66;
	or.b32 	%r77, %r76, 96;
	.loc	1 83 57                         // sk11_vision.py:83:57
	rem.s32 	%r78, %r73, %r18;
	rem.s32 	%r79, %r74, %r18;
	rem.s32 	%r80, %r75, %r18;
	rem.s32 	%r81, %r77, %r18;
	.loc	1 86 39                         // sk11_vision.py:86:39
	mul.lo.s32 	%r82, %r71, %r54;
	mul.lo.s32 	%r83, %r72, %r54;
	.loc	1 86 21                         // sk11_vision.py:86:21
	mad.wide.s32 	%rd30, %r82, 2, %rd8;
	mad.wide.s32 	%rd31, %r83, 2, %rd8;
	.loc	1 86 58                         // sk11_vision.py:86:58
	and.b32 	%r6, %r2, 7;
	shl.b32 	%r84, %r6, 3;
	.loc	1 86 51                         // sk11_vision.py:86:51
	mul.wide.u32 	%rd32, %r84, 2;
	add.s64 	%rd12, %rd30, %rd32;
	add.s64 	%rd13, %rd31, %rd32;
	.loc	1 87 21                         // sk11_vision.py:87:21
	add.s64 	%rd33, %rd9, %rd32;
	.loc	1 87 69                         // sk11_vision.py:87:69
	mul.lo.s32 	%r85, %r78, %r56;
	mul.lo.s32 	%r86, %r79, %r56;
	mul.lo.s32 	%r87, %r80, %r56;
	mul.lo.s32 	%r88, %r81, %r56;
	.loc	1 87 51                         // sk11_vision.py:87:51
	mad.wide.s32 	%rd14, %r85, 2, %rd33;
	mad.wide.s32 	%rd15, %r86, 2, %rd33;
	mad.wide.s32 	%rd16, %r87, 2, %rd33;
	mad.wide.s32 	%rd17, %r88, 2, %rd33;
$L__tmp4:
	.loc	2 43 17                         // standard.py:43:17 @[ sk11_vision.py:90:33 ]
	add.s32 	%r89, %r53, 63;
$L__tmp5:
	.loc	1 90 22                         // sk11_vision.py:90:22
	setp.lt.s32 	%p1, %r89, 64;
	setp.gt.s32 	%p2, %r89, 63;
	.loc	1 91 30                         // sk11_vision.py:91:30
	shl.b32 	%r8, %r2, 4;
	and.b32 	%r93, %r8, 4080;
	shl.b32 	%r9, %r2, 1;
	and.b32 	%r94, %r9, 112;
	xor.b32 	%r95, %r93, %r94;
	mov.b32 	%r96, global_smem;
	add.s32 	%r23, %r96, %r95;
	add.s32 	%r20, %r23, 49152;
	selp.b32 	%r21, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r20 + 0 ], [ %rd12 + 0 ], 0x10, %r21;
	// end inline asm
	add.s32 	%r22, %r23, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r22 + 0 ], [ %rd13 + 0 ], 0x10, %r21;
	// end inline asm
	cp.async.commit_group;
	.loc	1 91 47                         // sk11_vision.py:91:47
	// begin inline asm
	cp.async.cg.shared.global [ %r23 + 0 ], [ %rd14 + 0 ], 0x10, %r21;
	// end inline asm
	add.s32 	%r24, %r23, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r24 + 0 ], [ %rd15 + 0 ], 0x10, %r21;
	// end inline asm
	add.s32 	%r25, %r23, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r25 + 0 ], [ %rd16 + 0 ], 0x10, %r21;
	// end inline asm
	add.s32 	%r26, %r23, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r26 + 0 ], [ %rd17 + 0 ], 0x10, %r21;
	// end inline asm
	cp.async.commit_group;
	.loc	1 90 22                         // sk11_vision.py:90:22
	setp.gt.s32 	%p3, %r89, 127;
	.loc	1 92 18                         // sk11_vision.py:92:18
	add.s64 	%rd18, %rd12, 128;
	add.s64 	%rd19, %rd13, 128;
	.loc	1 93 18                         // sk11_vision.py:93:18
	add.s64 	%rd20, %rd14, 128;
	add.s64 	%rd21, %rd15, 128;
	add.s64 	%rd22, %rd16, 128;
	add.s64 	%rd23, %rd17, 128;
	.loc	1 91 30                         // sk11_vision.py:91:30
	bar.sync 	0;
	add.s32 	%r27, %r23, 57344;
	selp.b32 	%r28, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r27 + 0 ], [ %rd18 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r29, %r23, 61440;
	// begin inline asm
	cp.async.cg.shared.global [ %r29 + 0 ], [ %rd19 + 0 ], 0x10, %r28;
	// end inline asm
	cp.async.commit_group;
	.loc	1 91 47                         // sk11_vision.py:91:47
	add.s32 	%r30, %r23, 16384;
	// begin inline asm
	cp.async.cg.shared.global [ %r30 + 0 ], [ %rd20 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r31, %r23, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r31 + 0 ], [ %rd21 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r32, %r23, 24576;
	// begin inline asm
	cp.async.cg.shared.global [ %r32 + 0 ], [ %rd22 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r33, %r23, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r33 + 0 ], [ %rd23 + 0 ], 0x10, %r28;
	// end inline asm
	cp.async.commit_group;
	.loc	1 90 22                         // sk11_vision.py:90:22
	setp.gt.s32 	%p4, %r89, 191;
	.loc	1 92 18                         // sk11_vision.py:92:18
	add.s64 	%rd24, %rd12, 256;
	add.s64 	%rd25, %rd13, 256;
	.loc	1 93 18                         // sk11_vision.py:93:18
	add.s64 	%rd26, %rd14, 256;
	add.s64 	%rd27, %rd15, 256;
	add.s64 	%rd28, %rd16, 256;
	add.s64 	%rd29, %rd17, 256;
	.loc	1 91 30                         // sk11_vision.py:91:30
	bar.sync 	0;
	add.s32 	%r34, %r23, 65536;
	selp.b32 	%r35, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r34 + 0 ], [ %rd24 + 0 ], 0x10, %r35;
	// end inline asm
	add.s32 	%r36, %r23, 69632;
	// begin inline asm
	cp.async.cg.shared.global [ %r36 + 0 ], [ %rd25 + 0 ], 0x10, %r35;
	// end inline asm
	cp.async.commit_group;
	.loc	1 91 47                         // sk11_vision.py:91:47
	add.s32 	%r37, %r23, 32768;
	// begin inline asm
	cp.async.cg.shared.global [ %r37 + 0 ], [ %rd26 + 0 ], 0x10, %r35;
	// end inline asm
	add.s32 	%r38, %r23, 36864;
	// begin inline asm
	cp.async.cg.shared.global [ %r38 + 0 ], [ %rd27 + 0 ], 0x10, %r35;
	// end inline asm
	add.s32 	%r39, %r23, 40960;
	// begin inline asm
	cp.async.cg.shared.global [ %r39 + 0 ], [ %rd28 + 0 ], 0x10, %r35;
	// end inline asm
	add.s32 	%r40, %r23, 45056;
	// begin inline asm
	cp.async.cg.shared.global [ %r40 + 0 ], [ %rd29 + 0 ], 0x10, %r35;
	// end inline asm
	cp.async.commit_group;
	mov.b32 	%r730, 0f00000000;
	mov.b32 	%r731, %r730;
	mov.b32 	%r732, %r730;
	mov.b32 	%r733, %r730;
	mov.b32 	%r734, %r730;
	mov.b32 	%r735, %r730;
	mov.b32 	%r736, %r730;
	mov.b32 	%r737, %r730;
	mov.b32 	%r738, %r730;
	mov.b32 	%r739, %r730;
	mov.b32 	%r740, %r730;
	mov.b32 	%r741, %r730;
	mov.b32 	%r742, %r730;
	mov.b32 	%r743, %r730;
	mov.b32 	%r744, %r730;
	mov.b32 	%r745, %r730;
	mov.b32 	%r746, %r730;
	mov.b32 	%r747, %r730;
	mov.b32 	%r748, %r730;
	mov.b32 	%r749, %r730;
	mov.b32 	%r750, %r730;
	mov.b32 	%r751, %r730;
	mov.b32 	%r752, %r730;
	mov.b32 	%r753, %r730;
	mov.b32 	%r754, %r730;
	mov.b32 	%r755, %r730;
	mov.b32 	%r756, %r730;
	mov.b32 	%r757, %r730;
	mov.b32 	%r758, %r730;
	mov.b32 	%r759, %r730;
	mov.b32 	%r760, %r730;
	mov.b32 	%r761, %r730;
	.loc	1 90 22                         // sk11_vision.py:90:22
	@%p1 bra 	$L__BB0_3;
// %bb.1:                               // %.lr.ph
	.loc	1 0 22                          // sk11_vision.py:0:22
	cvt.s64.s32 	%rd1, %r82;
	cvt.s64.s32 	%rd2, %r83;
	cvt.s64.s32 	%rd3, %r85;
	cvt.s64.s32 	%rd4, %r86;
	cvt.s64.s32 	%rd5, %r87;
	cvt.s64.s32 	%rd6, %r88;
	shr.s32 	%r90, %r89, 31;
	shr.u32 	%r91, %r90, 26;
	add.s32 	%r92, %r89, %r91;
	shr.s32 	%r7, %r92, 6;
	add.s32 	%r10, %r7, -3;
	shl.b32 	%r97, %r5, 7;
	and.b32 	%r98, %r8, 2160;
	and.b32 	%r99, %r2, 16;
	or.b32 	%r100, %r97, %r98;
	xor.b32 	%r11, %r100, %r99;
	xor.b32 	%r12, %r11, 32;
	xor.b32 	%r13, %r11, 64;
	xor.b32 	%r14, %r11, 96;
	shl.b32 	%r101, %r6, 7;
	shl.b32 	%r102, %r4, 5;
	shl.b32 	%r103, %r6, 4;
	and.b32 	%r104, %r9, 48;
	or.b32 	%r105, %r101, %r102;
	xor.b32 	%r106, %r103, %r104;
	or.b32 	%r15, %r105, %r106;
	xor.b32 	%r16, %r15, 64;
	.loc	1 90 22                         // sk11_vision.py:90:22
	mul.wide.u32 	%rd7, %r6, 16;
	shl.b64 	%rd34, %rd6, 1;
	add.s64 	%rd35, %rd34, %rd9;
	add.s64 	%rd70, %rd35, 384;
	shl.b64 	%rd36, %rd5, 1;
	add.s64 	%rd37, %rd36, %rd9;
	add.s64 	%rd69, %rd37, 384;
	shl.b64 	%rd38, %rd4, 1;
	add.s64 	%rd39, %rd38, %rd9;
	add.s64 	%rd68, %rd39, 384;
	shl.b64 	%rd40, %rd3, 1;
	add.s64 	%rd41, %rd40, %rd9;
	add.s64 	%rd67, %rd41, 384;
	shl.b64 	%rd42, %rd2, 1;
	add.s64 	%rd43, %rd42, %rd8;
	add.s64 	%rd66, %rd43, 384;
	shl.b64 	%rd44, %rd1, 1;
	add.s64 	%rd45, %rd44, %rd8;
	add.s64 	%rd65, %rd45, 384;
	mov.b32 	%r762, 0;
	mov.b32 	%r730, 0f00000000;
	mov.b32 	%r729, 2;
	mov.b32 	%r728, -1;
	mov.b32 	%r731, %r730;
	mov.b32 	%r732, %r730;
	mov.b32 	%r733, %r730;
	mov.b32 	%r734, %r730;
	mov.b32 	%r735, %r730;
	mov.b32 	%r736, %r730;
	mov.b32 	%r737, %r730;
	mov.b32 	%r738, %r730;
	mov.b32 	%r739, %r730;
	mov.b32 	%r740, %r730;
	mov.b32 	%r741, %r730;
	mov.b32 	%r742, %r730;
	mov.b32 	%r743, %r730;
	mov.b32 	%r744, %r730;
	mov.b32 	%r745, %r730;
	mov.b32 	%r746, %r730;
	mov.b32 	%r747, %r730;
	mov.b32 	%r748, %r730;
	mov.b32 	%r749, %r730;
	mov.b32 	%r750, %r730;
	mov.b32 	%r751, %r730;
	mov.b32 	%r752, %r730;
	mov.b32 	%r753, %r730;
	mov.b32 	%r754, %r730;
	mov.b32 	%r755, %r730;
	mov.b32 	%r756, %r730;
	mov.b32 	%r757, %r730;
	mov.b32 	%r758, %r730;
	mov.b32 	%r759, %r730;
	mov.b32 	%r760, %r730;
	mov.b32 	%r761, %r730;
$L__BB0_2:                              // =>This Inner Loop Header: Depth=1
	setp.lt.s32 	%p5, %r762, %r10;
	add.s32 	%r178, %r728, 1;
	setp.gt.s32 	%p6, %r178, 2;
	selp.b32 	%r728, 0, %r178, %p6;
	.loc	1 91 30                         // sk11_vision.py:91:30
	cp.async.wait_group 	4;
	bar.sync 	0;
	shl.b32 	%r179, %r728, 13;
	add.s32 	%r180, %r96, %r179;
	add.s32 	%r181, %r180, %r11;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r107, %r108, %r109, %r110}, [%r181+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r119, %r120, %r121, %r122}, [%r181+53248];
	add.s32 	%r182, %r180, %r12;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r123, %r124, %r125, %r126}, [%r182+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r135, %r136, %r137, %r138}, [%r182+53248];
	add.s32 	%r183, %r180, %r13;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r139, %r140, %r141, %r142}, [%r183+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r151, %r152, %r153, %r154}, [%r183+53248];
	add.s32 	%r184, %r180, %r14;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r155, %r156, %r157, %r158}, [%r184+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r167, %r168, %r169, %r170}, [%r184+53248];
	.loc	1 91 47                         // sk11_vision.py:91:47
	add.s32 	%r185, %r180, %r179;
	add.s32 	%r186, %r185, %r15;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r111, %r112, %r127, %r128}, [%r186];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r113, %r114, %r129, %r130}, [%r186+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r115, %r116, %r131, %r132}, [%r186+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r117, %r118, %r133, %r134}, [%r186+12288];
	add.s32 	%r187, %r185, %r16;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r143, %r144, %r159, %r160}, [%r187];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r145, %r146, %r161, %r162}, [%r187+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r147, %r148, %r163, %r164}, [%r187+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r149, %r150, %r165, %r166}, [%r187+12288];
	.loc	1 91 39                         // sk11_vision.py:91:39
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r730, %r731, %r732, %r733 }, { %r107, %r108, %r109, %r110 }, { %r111, %r112 }, { %r730, %r731, %r732, %r733 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r734, %r735, %r736, %r737 }, { %r107, %r108, %r109, %r110 }, { %r113, %r114 }, { %r734, %r735, %r736, %r737 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r738, %r739, %r740, %r741 }, { %r107, %r108, %r109, %r110 }, { %r115, %r116 }, { %r738, %r739, %r740, %r741 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r742, %r743, %r744, %r745 }, { %r107, %r108, %r109, %r110 }, { %r117, %r118 }, { %r742, %r743, %r744, %r745 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r746, %r747, %r748, %r749 }, { %r119, %r120, %r121, %r122 }, { %r111, %r112 }, { %r746, %r747, %r748, %r749 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r750, %r751, %r752, %r753 }, { %r119, %r120, %r121, %r122 }, { %r113, %r114 }, { %r750, %r751, %r752, %r753 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r754, %r755, %r756, %r757 }, { %r119, %r120, %r121, %r122 }, { %r115, %r116 }, { %r754, %r755, %r756, %r757 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r758, %r759, %r760, %r761 }, { %r119, %r120, %r121, %r122 }, { %r117, %r118 }, { %r758, %r759, %r760, %r761 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r730, %r731, %r732, %r733 }, { %r123, %r124, %r125, %r126 }, { %r127, %r128 }, { %r730, %r731, %r732, %r733 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r734, %r735, %r736, %r737 }, { %r123, %r124, %r125, %r126 }, { %r129, %r130 }, { %r734, %r735, %r736, %r737 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r738, %r739, %r740, %r741 }, { %r123, %r124, %r125, %r126 }, { %r131, %r132 }, { %r738, %r739, %r740, %r741 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r742, %r743, %r744, %r745 }, { %r123, %r124, %r125, %r126 }, { %r133, %r134 }, { %r742, %r743, %r744, %r745 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r746, %r747, %r748, %r749 }, { %r135, %r136, %r137, %r138 }, { %r127, %r128 }, { %r746, %r747, %r748, %r749 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r750, %r751, %r752, %r753 }, { %r135, %r136, %r137, %r138 }, { %r129, %r130 }, { %r750, %r751, %r752, %r753 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r754, %r755, %r756, %r757 }, { %r135, %r136, %r137, %r138 }, { %r131, %r132 }, { %r754, %r755, %r756, %r757 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r758, %r759, %r760, %r761 }, { %r135, %r136, %r137, %r138 }, { %r133, %r134 }, { %r758, %r759, %r760, %r761 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r730, %r731, %r732, %r733 }, { %r139, %r140, %r141, %r142 }, { %r143, %r144 }, { %r730, %r731, %r732, %r733 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r734, %r735, %r736, %r737 }, { %r139, %r140, %r141, %r142 }, { %r145, %r146 }, { %r734, %r735, %r736, %r737 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r738, %r739, %r740, %r741 }, { %r139, %r140, %r141, %r142 }, { %r147, %r148 }, { %r738, %r739, %r740, %r741 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r742, %r743, %r744, %r745 }, { %r139, %r140, %r141, %r142 }, { %r149, %r150 }, { %r742, %r743, %r744, %r745 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r746, %r747, %r748, %r749 }, { %r151, %r152, %r153, %r154 }, { %r143, %r144 }, { %r746, %r747, %r748, %r749 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r750, %r751, %r752, %r753 }, { %r151, %r152, %r153, %r154 }, { %r145, %r146 }, { %r750, %r751, %r752, %r753 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r754, %r755, %r756, %r757 }, { %r151, %r152, %r153, %r154 }, { %r147, %r148 }, { %r754, %r755, %r756, %r757 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r758, %r759, %r760, %r761 }, { %r151, %r152, %r153, %r154 }, { %r149, %r150 }, { %r758, %r759, %r760, %r761 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r730, %r731, %r732, %r733 }, { %r155, %r156, %r157, %r158 }, { %r159, %r160 }, { %r730, %r731, %r732, %r733 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r734, %r735, %r736, %r737 }, { %r155, %r156, %r157, %r158 }, { %r161, %r162 }, { %r734, %r735, %r736, %r737 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r738, %r739, %r740, %r741 }, { %r155, %r156, %r157, %r158 }, { %r163, %r164 }, { %r738, %r739, %r740, %r741 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r742, %r743, %r744, %r745 }, { %r155, %r156, %r157, %r158 }, { %r165, %r166 }, { %r742, %r743, %r744, %r745 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r746, %r747, %r748, %r749 }, { %r167, %r168, %r169, %r170 }, { %r159, %r160 }, { %r746, %r747, %r748, %r749 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r750, %r751, %r752, %r753 }, { %r167, %r168, %r169, %r170 }, { %r161, %r162 }, { %r750, %r751, %r752, %r753 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r754, %r755, %r756, %r757 }, { %r167, %r168, %r169, %r170 }, { %r163, %r164 }, { %r754, %r755, %r756, %r757 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r758, %r759, %r760, %r761 }, { %r167, %r168, %r169, %r170 }, { %r165, %r166 }, { %r758, %r759, %r760, %r761 };
	// end inline asm
	.loc	1 92 18                         // sk11_vision.py:92:18
	add.s64 	%rd46, %rd65, %rd7;
	.loc	1 93 18                         // sk11_vision.py:93:18
	add.s64 	%rd47, %rd66, %rd7;
	add.s64 	%rd48, %rd67, %rd7;
	add.s64 	%rd49, %rd68, %rd7;
	add.s64 	%rd50, %rd69, %rd7;
	.loc	1 90 22                         // sk11_vision.py:90:22
	add.s64 	%rd51, %rd70, %rd7;
	add.s32 	%r188, %r729, 1;
	setp.gt.s32 	%p7, %r188, 2;
	selp.b32 	%r729, 0, %r188, %p7;
	.loc	1 91 30                         // sk11_vision.py:91:30
	shl.b32 	%r189, %r729, 13;
	bar.sync 	0;
	add.s32 	%r190, %r23, %r189;
	add.s32 	%r171, %r190, 49152;
	selp.b32 	%r172, 16, 0, %p5;
	// begin inline asm
	cp.async.cg.shared.global [ %r171 + 0 ], [ %rd46 + 0 ], 0x10, %r172;
	// end inline asm
	add.s32 	%r173, %r190, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r173 + 0 ], [ %rd47 + 0 ], 0x10, %r172;
	// end inline asm
	cp.async.commit_group;
	.loc	1 91 47                         // sk11_vision.py:91:47
	add.s32 	%r174, %r190, %r189;
	// begin inline asm
	cp.async.cg.shared.global [ %r174 + 0 ], [ %rd48 + 0 ], 0x10, %r172;
	// end inline asm
	add.s32 	%r175, %r174, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r175 + 0 ], [ %rd49 + 0 ], 0x10, %r172;
	// end inline asm
	add.s32 	%r176, %r174, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r176 + 0 ], [ %rd50 + 0 ], 0x10, %r172;
	// end inline asm
	add.s32 	%r177, %r174, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r177 + 0 ], [ %rd51 + 0 ], 0x10, %r172;
	// end inline asm
	cp.async.commit_group;
	.loc	1 90 22                         // sk11_vision.py:90:22
	add.s32 	%r762, %r762, 1;
	add.s64 	%rd70, %rd70, 128;
	add.s64 	%rd69, %rd69, 128;
	add.s64 	%rd68, %rd68, 128;
	add.s64 	%rd67, %rd67, 128;
	add.s64 	%rd66, %rd66, 128;
	add.s64 	%rd65, %rd65, 128;
	setp.ne.b32 	%p8, %r7, %r762;
	@%p8 bra 	$L__BB0_2;
$L__BB0_3:                              // %._crit_edge
	.loc	1 83 45                         // sk11_vision.py:83:45
	and.b32 	%r219, %r2, 3;
	shl.b32 	%r220, %r219, 1;
	shr.u32 	%r221, %r4, 2;
	or.b32 	%r222, %r220, %r221;
	.loc	1 83 32                         // sk11_vision.py:83:32
	or.b32 	%r223, %r222, %r3;
	or.b32 	%r224, %r223, 96;
	.loc	1 83 57                         // sk11_vision.py:83:57
	rem.s32 	%r225, %r224, %r18;
	.loc	1 83 32                         // sk11_vision.py:83:32
	or.b32 	%r226, %r223, 64;
	.loc	1 83 57                         // sk11_vision.py:83:57
	rem.s32 	%r227, %r226, %r18;
	.loc	1 83 32                         // sk11_vision.py:83:32
	or.b32 	%r228, %r223, 32;
	.loc	1 83 57                         // sk11_vision.py:83:57
	rem.s32 	%r229, %r228, %r18;
	rem.s32 	%r230, %r223, %r18;
	.loc	1 83 45                         // sk11_vision.py:83:45
	shl.b32 	%r231, %r5, 3;
	.loc	1 83 32                         // sk11_vision.py:83:32
	or.b32 	%r232, %r3, %r231;
	.loc	1 82 45                         // sk11_vision.py:82:45
	shr.u32 	%r233, %r2, 4;
	.loc	1 82 32                         // sk11_vision.py:82:32
	or.b32 	%r234, %r233, %r1;
	or.b32 	%r235, %r234, 48;
	.loc	1 82 45                         // sk11_vision.py:82:45
	bfe.u32 	%r236, %r2, 4, 4;
	.loc	1 82 32                         // sk11_vision.py:82:32
	or.b32 	%r237, %r236, %r1;
	or.b32 	%r238, %r237, 32;
	or.b32 	%r239, %r237, 16;
	.loc	1 90 22                         // sk11_vision.py:90:22
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 95 33                         // sk11_vision.py:95:33
	mad.wide.s32 	%rd52, %r230, 2, %rd10;
	mad.wide.s32 	%rd53, %r229, 2, %rd10;
	mad.wide.s32 	%rd54, %r227, 2, %rd10;
	mad.wide.s32 	%rd55, %r225, 2, %rd10;
	.loc	1 95 22                         // sk11_vision.py:95:22
	// begin inline asm
	mov.u32 %r191, 0x0;
	ld.global.b32 { %r191 }, [ %rd52 + 0 ];
	// end inline asm
	mov.b32 	{%rs33, %rs34}, %r191;
	// begin inline asm
	mov.u32 %r192, 0x0;
	ld.global.b32 { %r192 }, [ %rd53 + 0 ];
	// end inline asm
	mov.b32 	{%rs35, %rs36}, %r192;
	// begin inline asm
	mov.u32 %r193, 0x0;
	ld.global.b32 { %r193 }, [ %rd54 + 0 ];
	// end inline asm
	mov.b32 	{%rs37, %rs38}, %r193;
	// begin inline asm
	mov.u32 %r194, 0x0;
	ld.global.b32 { %r194 }, [ %rd55 + 0 ];
	// end inline asm
	mov.b32 	{%rs39, %rs40}, %r194;
	.loc	1 95 58                         // sk11_vision.py:95:58
	cvt.f32.bf16 	%r240, %rs33;
	cvt.f32.bf16 	%r241, %rs34;
	cvt.f32.bf16 	%r242, %rs35;
	cvt.f32.bf16 	%r243, %rs36;
	cvt.f32.bf16 	%r244, %rs37;
	cvt.f32.bf16 	%r245, %rs38;
	cvt.f32.bf16 	%r246, %rs39;
	cvt.f32.bf16 	%r247, %rs40;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r248, %r730, %r240;
	add.f32 	%r249, %r731, %r241;
	add.f32 	%r250, %r732, %r240;
	add.f32 	%r251, %r733, %r241;
	add.f32 	%r252, %r734, %r242;
	add.f32 	%r253, %r735, %r243;
	add.f32 	%r254, %r736, %r242;
	add.f32 	%r255, %r737, %r243;
	add.f32 	%r256, %r738, %r244;
	add.f32 	%r257, %r739, %r245;
	add.f32 	%r258, %r740, %r244;
	add.f32 	%r259, %r741, %r245;
	add.f32 	%r260, %r742, %r246;
	add.f32 	%r261, %r743, %r247;
	add.f32 	%r262, %r744, %r246;
	add.f32 	%r263, %r745, %r247;
	add.f32 	%r264, %r746, %r240;
	add.f32 	%r265, %r747, %r241;
	add.f32 	%r266, %r748, %r240;
	add.f32 	%r267, %r749, %r241;
	add.f32 	%r268, %r750, %r242;
	add.f32 	%r269, %r751, %r243;
	add.f32 	%r270, %r752, %r242;
	add.f32 	%r271, %r753, %r243;
	add.f32 	%r272, %r754, %r244;
	add.f32 	%r273, %r755, %r245;
	add.f32 	%r274, %r756, %r244;
	add.f32 	%r275, %r757, %r245;
	add.f32 	%r276, %r758, %r246;
	add.f32 	%r277, %r759, %r247;
	add.f32 	%r278, %r760, %r246;
	add.f32 	%r279, %r761, %r247;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r280, %r248, 0f3D372713;
	mul.f32 	%r281, %r249, 0f3D372713;
	mul.f32 	%r282, %r250, 0f3D372713;
	mul.f32 	%r283, %r251, 0f3D372713;
	mul.f32 	%r284, %r252, 0f3D372713;
	mul.f32 	%r285, %r253, 0f3D372713;
	mul.f32 	%r286, %r254, 0f3D372713;
	mul.f32 	%r287, %r255, 0f3D372713;
	mul.f32 	%r288, %r256, 0f3D372713;
	mul.f32 	%r289, %r257, 0f3D372713;
	mul.f32 	%r290, %r258, 0f3D372713;
	mul.f32 	%r291, %r259, 0f3D372713;
	mul.f32 	%r292, %r260, 0f3D372713;
	mul.f32 	%r293, %r261, 0f3D372713;
	mul.f32 	%r294, %r262, 0f3D372713;
	mul.f32 	%r295, %r263, 0f3D372713;
	mul.f32 	%r296, %r264, 0f3D372713;
	mul.f32 	%r297, %r265, 0f3D372713;
	mul.f32 	%r298, %r266, 0f3D372713;
	mul.f32 	%r299, %r267, 0f3D372713;
	mul.f32 	%r300, %r268, 0f3D372713;
	mul.f32 	%r301, %r269, 0f3D372713;
	mul.f32 	%r302, %r270, 0f3D372713;
	mul.f32 	%r303, %r271, 0f3D372713;
	mul.f32 	%r304, %r272, 0f3D372713;
	mul.f32 	%r305, %r273, 0f3D372713;
	mul.f32 	%r306, %r274, 0f3D372713;
	mul.f32 	%r307, %r275, 0f3D372713;
	mul.f32 	%r308, %r276, 0f3D372713;
	mul.f32 	%r309, %r277, 0f3D372713;
	mul.f32 	%r310, %r278, 0f3D372713;
	mul.f32 	%r311, %r279, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r312, %r248, %r280;
	mul.f32 	%r313, %r249, %r281;
	mul.f32 	%r314, %r250, %r282;
	mul.f32 	%r315, %r251, %r283;
	mul.f32 	%r316, %r252, %r284;
	mul.f32 	%r317, %r253, %r285;
	mul.f32 	%r318, %r254, %r286;
	mul.f32 	%r319, %r255, %r287;
	mul.f32 	%r320, %r256, %r288;
	mul.f32 	%r321, %r257, %r289;
	mul.f32 	%r322, %r258, %r290;
	mul.f32 	%r323, %r259, %r291;
	mul.f32 	%r324, %r260, %r292;
	mul.f32 	%r325, %r261, %r293;
	mul.f32 	%r326, %r262, %r294;
	mul.f32 	%r327, %r263, %r295;
	mul.f32 	%r328, %r264, %r296;
	mul.f32 	%r329, %r265, %r297;
	mul.f32 	%r330, %r266, %r298;
	mul.f32 	%r331, %r267, %r299;
	mul.f32 	%r332, %r268, %r300;
	mul.f32 	%r333, %r269, %r301;
	mul.f32 	%r334, %r270, %r302;
	mul.f32 	%r335, %r271, %r303;
	mul.f32 	%r336, %r272, %r304;
	mul.f32 	%r337, %r273, %r305;
	mul.f32 	%r338, %r274, %r306;
	mul.f32 	%r339, %r275, %r307;
	mul.f32 	%r340, %r276, %r308;
	mul.f32 	%r341, %r277, %r309;
	mul.f32 	%r342, %r278, %r310;
	mul.f32 	%r343, %r279, %r311;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r344, %r248, %r312, %r248;
	fma.rn.f32 	%r345, %r249, %r313, %r249;
	fma.rn.f32 	%r346, %r250, %r314, %r250;
	fma.rn.f32 	%r347, %r251, %r315, %r251;
	fma.rn.f32 	%r348, %r252, %r316, %r252;
	fma.rn.f32 	%r349, %r253, %r317, %r253;
	fma.rn.f32 	%r350, %r254, %r318, %r254;
	fma.rn.f32 	%r351, %r255, %r319, %r255;
	fma.rn.f32 	%r352, %r256, %r320, %r256;
	fma.rn.f32 	%r353, %r257, %r321, %r257;
	fma.rn.f32 	%r354, %r258, %r322, %r258;
	fma.rn.f32 	%r355, %r259, %r323, %r259;
	fma.rn.f32 	%r356, %r260, %r324, %r260;
	fma.rn.f32 	%r357, %r261, %r325, %r261;
	fma.rn.f32 	%r358, %r262, %r326, %r262;
	fma.rn.f32 	%r359, %r263, %r327, %r263;
	fma.rn.f32 	%r360, %r264, %r328, %r264;
	fma.rn.f32 	%r361, %r265, %r329, %r265;
	fma.rn.f32 	%r362, %r266, %r330, %r266;
	fma.rn.f32 	%r363, %r267, %r331, %r267;
	fma.rn.f32 	%r364, %r268, %r332, %r268;
	fma.rn.f32 	%r365, %r269, %r333, %r269;
	fma.rn.f32 	%r366, %r270, %r334, %r270;
	fma.rn.f32 	%r367, %r271, %r335, %r271;
	fma.rn.f32 	%r368, %r272, %r336, %r272;
	fma.rn.f32 	%r369, %r273, %r337, %r273;
	fma.rn.f32 	%r370, %r274, %r338, %r274;
	fma.rn.f32 	%r371, %r275, %r339, %r275;
	fma.rn.f32 	%r372, %r276, %r340, %r276;
	fma.rn.f32 	%r373, %r277, %r341, %r277;
	fma.rn.f32 	%r374, %r278, %r342, %r278;
	fma.rn.f32 	%r375, %r279, %r343, %r279;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r376, %r344, 0f3F4C422A;
	mul.f32 	%r377, %r345, 0f3F4C422A;
	mul.f32 	%r378, %r346, 0f3F4C422A;
	mul.f32 	%r379, %r347, 0f3F4C422A;
	mul.f32 	%r380, %r348, 0f3F4C422A;
	mul.f32 	%r381, %r349, 0f3F4C422A;
	mul.f32 	%r382, %r350, 0f3F4C422A;
	mul.f32 	%r383, %r351, 0f3F4C422A;
	mul.f32 	%r384, %r352, 0f3F4C422A;
	mul.f32 	%r385, %r353, 0f3F4C422A;
	mul.f32 	%r386, %r354, 0f3F4C422A;
	mul.f32 	%r387, %r355, 0f3F4C422A;
	mul.f32 	%r388, %r356, 0f3F4C422A;
	mul.f32 	%r389, %r357, 0f3F4C422A;
	mul.f32 	%r390, %r358, 0f3F4C422A;
	mul.f32 	%r391, %r359, 0f3F4C422A;
	mul.f32 	%r392, %r360, 0f3F4C422A;
	mul.f32 	%r393, %r361, 0f3F4C422A;
	mul.f32 	%r394, %r362, 0f3F4C422A;
	mul.f32 	%r395, %r363, 0f3F4C422A;
	mul.f32 	%r396, %r364, 0f3F4C422A;
	mul.f32 	%r397, %r365, 0f3F4C422A;
	mul.f32 	%r398, %r366, 0f3F4C422A;
	mul.f32 	%r399, %r367, 0f3F4C422A;
	mul.f32 	%r400, %r368, 0f3F4C422A;
	mul.f32 	%r401, %r369, 0f3F4C422A;
	mul.f32 	%r402, %r370, 0f3F4C422A;
	mul.f32 	%r403, %r371, 0f3F4C422A;
	mul.f32 	%r404, %r372, 0f3F4C422A;
	mul.f32 	%r405, %r373, 0f3F4C422A;
	mul.f32 	%r406, %r374, 0f3F4C422A;
	mul.f32 	%r407, %r375, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r408, %r248, 0f3F000000;
	mul.f32 	%r409, %r249, 0f3F000000;
	mul.f32 	%r410, %r250, 0f3F000000;
	mul.f32 	%r411, %r251, 0f3F000000;
	mul.f32 	%r412, %r252, 0f3F000000;
	mul.f32 	%r413, %r253, 0f3F000000;
	mul.f32 	%r414, %r254, 0f3F000000;
	mul.f32 	%r415, %r255, 0f3F000000;
	mul.f32 	%r416, %r256, 0f3F000000;
	mul.f32 	%r417, %r257, 0f3F000000;
	mul.f32 	%r418, %r258, 0f3F000000;
	mul.f32 	%r419, %r259, 0f3F000000;
	mul.f32 	%r420, %r260, 0f3F000000;
	mul.f32 	%r421, %r261, 0f3F000000;
	mul.f32 	%r422, %r262, 0f3F000000;
	mul.f32 	%r423, %r263, 0f3F000000;
	mul.f32 	%r424, %r264, 0f3F000000;
	mul.f32 	%r425, %r265, 0f3F000000;
	mul.f32 	%r426, %r266, 0f3F000000;
	mul.f32 	%r427, %r267, 0f3F000000;
	mul.f32 	%r428, %r268, 0f3F000000;
	mul.f32 	%r429, %r269, 0f3F000000;
	mul.f32 	%r430, %r270, 0f3F000000;
	mul.f32 	%r431, %r271, 0f3F000000;
	mul.f32 	%r432, %r272, 0f3F000000;
	mul.f32 	%r433, %r273, 0f3F000000;
	mul.f32 	%r434, %r274, 0f3F000000;
	mul.f32 	%r435, %r275, 0f3F000000;
	mul.f32 	%r436, %r276, 0f3F000000;
	mul.f32 	%r437, %r277, 0f3F000000;
	mul.f32 	%r438, %r278, 0f3F000000;
	mul.f32 	%r439, %r279, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r440, %r376, 0fC0000000;
	mul.f32 	%r441, %r377, 0fC0000000;
	mul.f32 	%r442, %r378, 0fC0000000;
	mul.f32 	%r443, %r379, 0fC0000000;
	mul.f32 	%r444, %r380, 0fC0000000;
	mul.f32 	%r445, %r381, 0fC0000000;
	mul.f32 	%r446, %r382, 0fC0000000;
	mul.f32 	%r447, %r383, 0fC0000000;
	mul.f32 	%r448, %r384, 0fC0000000;
	mul.f32 	%r449, %r385, 0fC0000000;
	mul.f32 	%r450, %r386, 0fC0000000;
	mul.f32 	%r451, %r387, 0fC0000000;
	mul.f32 	%r452, %r388, 0fC0000000;
	mul.f32 	%r453, %r389, 0fC0000000;
	mul.f32 	%r454, %r390, 0fC0000000;
	mul.f32 	%r455, %r391, 0fC0000000;
	mul.f32 	%r456, %r392, 0fC0000000;
	mul.f32 	%r457, %r393, 0fC0000000;
	mul.f32 	%r458, %r394, 0fC0000000;
	mul.f32 	%r459, %r395, 0fC0000000;
	mul.f32 	%r460, %r396, 0fC0000000;
	mul.f32 	%r461, %r397, 0fC0000000;
	mul.f32 	%r462, %r398, 0fC0000000;
	mul.f32 	%r463, %r399, 0fC0000000;
	mul.f32 	%r464, %r400, 0fC0000000;
	mul.f32 	%r465, %r401, 0fC0000000;
	mul.f32 	%r466, %r402, 0fC0000000;
	mul.f32 	%r467, %r403, 0fC0000000;
	mul.f32 	%r468, %r404, 0fC0000000;
	mul.f32 	%r469, %r405, 0fC0000000;
	mul.f32 	%r470, %r406, 0fC0000000;
	mul.f32 	%r471, %r407, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r472, %r440, 0f3FB8AA3B;
	ex2.approx.f32 	%r473, %r472;
	mul.f32 	%r474, %r441, 0f3FB8AA3B;
	ex2.approx.f32 	%r475, %r474;
	mul.f32 	%r476, %r442, 0f3FB8AA3B;
	ex2.approx.f32 	%r477, %r476;
	mul.f32 	%r478, %r443, 0f3FB8AA3B;
	ex2.approx.f32 	%r479, %r478;
	mul.f32 	%r480, %r444, 0f3FB8AA3B;
	ex2.approx.f32 	%r481, %r480;
	mul.f32 	%r482, %r445, 0f3FB8AA3B;
	ex2.approx.f32 	%r483, %r482;
	mul.f32 	%r484, %r446, 0f3FB8AA3B;
	ex2.approx.f32 	%r485, %r484;
	mul.f32 	%r486, %r447, 0f3FB8AA3B;
	ex2.approx.f32 	%r487, %r486;
	mul.f32 	%r488, %r448, 0f3FB8AA3B;
	ex2.approx.f32 	%r489, %r488;
	mul.f32 	%r490, %r449, 0f3FB8AA3B;
	ex2.approx.f32 	%r491, %r490;
	mul.f32 	%r492, %r450, 0f3FB8AA3B;
	ex2.approx.f32 	%r493, %r492;
	mul.f32 	%r494, %r451, 0f3FB8AA3B;
	ex2.approx.f32 	%r495, %r494;
	mul.f32 	%r496, %r452, 0f3FB8AA3B;
	ex2.approx.f32 	%r497, %r496;
	mul.f32 	%r498, %r453, 0f3FB8AA3B;
	ex2.approx.f32 	%r499, %r498;
	mul.f32 	%r500, %r454, 0f3FB8AA3B;
	ex2.approx.f32 	%r501, %r500;
	mul.f32 	%r502, %r455, 0f3FB8AA3B;
	ex2.approx.f32 	%r503, %r502;
	mul.f32 	%r504, %r456, 0f3FB8AA3B;
	ex2.approx.f32 	%r505, %r504;
	mul.f32 	%r506, %r457, 0f3FB8AA3B;
	ex2.approx.f32 	%r507, %r506;
	mul.f32 	%r508, %r458, 0f3FB8AA3B;
	ex2.approx.f32 	%r509, %r508;
	mul.f32 	%r510, %r459, 0f3FB8AA3B;
	ex2.approx.f32 	%r511, %r510;
	mul.f32 	%r512, %r460, 0f3FB8AA3B;
	ex2.approx.f32 	%r513, %r512;
	mul.f32 	%r514, %r461, 0f3FB8AA3B;
	ex2.approx.f32 	%r515, %r514;
	mul.f32 	%r516, %r462, 0f3FB8AA3B;
	ex2.approx.f32 	%r517, %r516;
	mul.f32 	%r518, %r463, 0f3FB8AA3B;
	ex2.approx.f32 	%r519, %r518;
	mul.f32 	%r520, %r464, 0f3FB8AA3B;
	ex2.approx.f32 	%r521, %r520;
	mul.f32 	%r522, %r465, 0f3FB8AA3B;
	ex2.approx.f32 	%r523, %r522;
	mul.f32 	%r524, %r466, 0f3FB8AA3B;
	ex2.approx.f32 	%r525, %r524;
	mul.f32 	%r526, %r467, 0f3FB8AA3B;
	ex2.approx.f32 	%r527, %r526;
	mul.f32 	%r528, %r468, 0f3FB8AA3B;
	ex2.approx.f32 	%r529, %r528;
	mul.f32 	%r530, %r469, 0f3FB8AA3B;
	ex2.approx.f32 	%r531, %r530;
	mul.f32 	%r532, %r470, 0f3FB8AA3B;
	ex2.approx.f32 	%r533, %r532;
	mul.f32 	%r534, %r471, 0f3FB8AA3B;
	ex2.approx.f32 	%r535, %r534;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r536, %r473, 0f3F800000;
	add.f32 	%r537, %r475, 0f3F800000;
	add.f32 	%r538, %r477, 0f3F800000;
	add.f32 	%r539, %r479, 0f3F800000;
	add.f32 	%r540, %r481, 0f3F800000;
	add.f32 	%r541, %r483, 0f3F800000;
	add.f32 	%r542, %r485, 0f3F800000;
	add.f32 	%r543, %r487, 0f3F800000;
	add.f32 	%r544, %r489, 0f3F800000;
	add.f32 	%r545, %r491, 0f3F800000;
	add.f32 	%r546, %r493, 0f3F800000;
	add.f32 	%r547, %r495, 0f3F800000;
	add.f32 	%r548, %r497, 0f3F800000;
	add.f32 	%r549, %r499, 0f3F800000;
	add.f32 	%r550, %r501, 0f3F800000;
	add.f32 	%r551, %r503, 0f3F800000;
	add.f32 	%r552, %r505, 0f3F800000;
	add.f32 	%r553, %r507, 0f3F800000;
	add.f32 	%r554, %r509, 0f3F800000;
	add.f32 	%r555, %r511, 0f3F800000;
	add.f32 	%r556, %r513, 0f3F800000;
	add.f32 	%r557, %r515, 0f3F800000;
	add.f32 	%r558, %r517, 0f3F800000;
	add.f32 	%r559, %r519, 0f3F800000;
	add.f32 	%r560, %r521, 0f3F800000;
	add.f32 	%r561, %r523, 0f3F800000;
	add.f32 	%r562, %r525, 0f3F800000;
	add.f32 	%r563, %r527, 0f3F800000;
	add.f32 	%r564, %r529, 0f3F800000;
	add.f32 	%r565, %r531, 0f3F800000;
	add.f32 	%r566, %r533, 0f3F800000;
	add.f32 	%r567, %r535, 0f3F800000;
	mov.b32 	%r568, 0f40000000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r569, %r568, %r536;
	div.full.f32 	%r570, %r568, %r537;
	div.full.f32 	%r571, %r568, %r538;
	div.full.f32 	%r572, %r568, %r539;
	div.full.f32 	%r573, %r568, %r540;
	div.full.f32 	%r574, %r568, %r541;
	div.full.f32 	%r575, %r568, %r542;
	div.full.f32 	%r576, %r568, %r543;
	div.full.f32 	%r577, %r568, %r544;
	div.full.f32 	%r578, %r568, %r545;
	div.full.f32 	%r579, %r568, %r546;
	div.full.f32 	%r580, %r568, %r547;
	div.full.f32 	%r581, %r568, %r548;
	div.full.f32 	%r582, %r568, %r549;
	div.full.f32 	%r583, %r568, %r550;
	div.full.f32 	%r584, %r568, %r551;
	div.full.f32 	%r585, %r568, %r552;
	div.full.f32 	%r586, %r568, %r553;
	div.full.f32 	%r587, %r568, %r554;
	div.full.f32 	%r588, %r568, %r555;
	div.full.f32 	%r589, %r568, %r556;
	div.full.f32 	%r590, %r568, %r557;
	div.full.f32 	%r591, %r568, %r558;
	div.full.f32 	%r592, %r568, %r559;
	div.full.f32 	%r593, %r568, %r560;
	div.full.f32 	%r594, %r568, %r561;
	div.full.f32 	%r595, %r568, %r562;
	div.full.f32 	%r596, %r568, %r563;
	div.full.f32 	%r597, %r568, %r564;
	div.full.f32 	%r598, %r568, %r565;
	div.full.f32 	%r599, %r568, %r566;
	div.full.f32 	%r600, %r568, %r567;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r601, %r569, 0fBF800000;
	add.f32 	%r602, %r570, 0fBF800000;
	add.f32 	%r603, %r571, 0fBF800000;
	add.f32 	%r604, %r572, 0fBF800000;
	add.f32 	%r605, %r573, 0fBF800000;
	add.f32 	%r606, %r574, 0fBF800000;
	add.f32 	%r607, %r575, 0fBF800000;
	add.f32 	%r608, %r576, 0fBF800000;
	add.f32 	%r609, %r577, 0fBF800000;
	add.f32 	%r610, %r578, 0fBF800000;
	add.f32 	%r611, %r579, 0fBF800000;
	add.f32 	%r612, %r580, 0fBF800000;
	add.f32 	%r613, %r581, 0fBF800000;
	add.f32 	%r614, %r582, 0fBF800000;
	add.f32 	%r615, %r583, 0fBF800000;
	add.f32 	%r616, %r584, 0fBF800000;
	add.f32 	%r617, %r585, 0fBF800000;
	add.f32 	%r618, %r586, 0fBF800000;
	add.f32 	%r619, %r587, 0fBF800000;
	add.f32 	%r620, %r588, 0fBF800000;
	add.f32 	%r621, %r589, 0fBF800000;
	add.f32 	%r622, %r590, 0fBF800000;
	add.f32 	%r623, %r591, 0fBF800000;
	add.f32 	%r624, %r592, 0fBF800000;
	add.f32 	%r625, %r593, 0fBF800000;
	add.f32 	%r626, %r594, 0fBF800000;
	add.f32 	%r627, %r595, 0fBF800000;
	add.f32 	%r628, %r596, 0fBF800000;
	add.f32 	%r629, %r597, 0fBF800000;
	add.f32 	%r630, %r598, 0fBF800000;
	add.f32 	%r631, %r599, 0fBF800000;
	add.f32 	%r632, %r600, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r633, %r601, 0f3F800000;
	add.f32 	%r634, %r602, 0f3F800000;
	add.f32 	%r635, %r603, 0f3F800000;
	add.f32 	%r636, %r604, 0f3F800000;
	add.f32 	%r637, %r605, 0f3F800000;
	add.f32 	%r638, %r606, 0f3F800000;
	add.f32 	%r639, %r607, 0f3F800000;
	add.f32 	%r640, %r608, 0f3F800000;
	add.f32 	%r641, %r609, 0f3F800000;
	add.f32 	%r642, %r610, 0f3F800000;
	add.f32 	%r643, %r611, 0f3F800000;
	add.f32 	%r644, %r612, 0f3F800000;
	add.f32 	%r645, %r613, 0f3F800000;
	add.f32 	%r646, %r614, 0f3F800000;
	add.f32 	%r647, %r615, 0f3F800000;
	add.f32 	%r648, %r616, 0f3F800000;
	add.f32 	%r649, %r617, 0f3F800000;
	add.f32 	%r650, %r618, 0f3F800000;
	add.f32 	%r651, %r619, 0f3F800000;
	add.f32 	%r652, %r620, 0f3F800000;
	add.f32 	%r653, %r621, 0f3F800000;
	add.f32 	%r654, %r622, 0f3F800000;
	add.f32 	%r655, %r623, 0f3F800000;
	add.f32 	%r656, %r624, 0f3F800000;
	add.f32 	%r657, %r625, 0f3F800000;
	add.f32 	%r658, %r626, 0f3F800000;
	add.f32 	%r659, %r627, 0f3F800000;
	add.f32 	%r660, %r628, 0f3F800000;
	add.f32 	%r661, %r629, 0f3F800000;
	add.f32 	%r662, %r630, 0f3F800000;
	add.f32 	%r663, %r631, 0f3F800000;
	add.f32 	%r664, %r632, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r665, %r408, %r633;
	mul.f32 	%r666, %r409, %r634;
	mul.f32 	%r667, %r410, %r635;
	mul.f32 	%r668, %r411, %r636;
	mul.f32 	%r669, %r412, %r637;
	mul.f32 	%r670, %r413, %r638;
	mul.f32 	%r671, %r414, %r639;
	mul.f32 	%r672, %r415, %r640;
	mul.f32 	%r673, %r416, %r641;
	mul.f32 	%r674, %r417, %r642;
	mul.f32 	%r675, %r418, %r643;
	mul.f32 	%r676, %r419, %r644;
	mul.f32 	%r677, %r420, %r645;
	mul.f32 	%r678, %r421, %r646;
	mul.f32 	%r679, %r422, %r647;
	mul.f32 	%r680, %r423, %r648;
	mul.f32 	%r681, %r424, %r649;
	mul.f32 	%r682, %r425, %r650;
	mul.f32 	%r683, %r426, %r651;
	mul.f32 	%r684, %r427, %r652;
	mul.f32 	%r685, %r428, %r653;
	mul.f32 	%r686, %r429, %r654;
	mul.f32 	%r687, %r430, %r655;
	mul.f32 	%r688, %r431, %r656;
	mul.f32 	%r689, %r432, %r657;
	mul.f32 	%r690, %r433, %r658;
	mul.f32 	%r691, %r434, %r659;
	mul.f32 	%r692, %r435, %r660;
	mul.f32 	%r693, %r436, %r661;
	mul.f32 	%r694, %r437, %r662;
	mul.f32 	%r695, %r438, %r663;
	mul.f32 	%r696, %r439, %r664;
	.loc	1 104 31                        // sk11_vision.py:104:31
	setp.lt.s32 	%p13, %r237, %r17;
	setp.lt.s32 	%p14, %r239, %r17;
	setp.lt.s32 	%p15, %r238, %r17;
	setp.lt.s32 	%p16, %r235, %r17;
	.loc	1 104 54                        // sk11_vision.py:104:54
	setp.lt.s32 	%p17, %r232, %r18;
	.loc	1 104 37                        // sk11_vision.py:104:37
	and.pred 	%p9, %p13, %p17;
	and.pred 	%p10, %p14, %p17;
	and.pred 	%p11, %p15, %p17;
	and.pred 	%p12, %p16, %p17;
	.loc	1 102 35                        // sk11_vision.py:102:35
	mul.lo.s32 	%r697, %r237, %r19;
	mul.lo.s32 	%r698, %r239, %r19;
	mul.lo.s32 	%r699, %r238, %r19;
	mul.lo.s32 	%r700, %r235, %r19;
	.loc	1 102 18                        // sk11_vision.py:102:18
	mad.wide.s32 	%rd60, %r697, 2, %rd11;
	mad.wide.s32 	%rd61, %r698, 2, %rd11;
	mad.wide.s32 	%rd62, %r699, 2, %rd11;
	mad.wide.s32 	%rd63, %r700, 2, %rd11;
	.loc	1 102 50                        // sk11_vision.py:102:50
	mul.wide.s32 	%rd64, %r232, 2;
	add.s64 	%rd56, %rd60, %rd64;
	add.s64 	%rd57, %rd61, %rd64;
	add.s64 	%rd58, %rd62, %rd64;
	add.s64 	%rd59, %rd63, %rd64;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16.f32 	%rs1, %r665;
	cvt.rn.bf16.f32 	%rs2, %r666;
	cvt.rn.bf16.f32 	%rs5, %r667;
	cvt.rn.bf16.f32 	%rs6, %r668;
	cvt.rn.bf16.f32 	%rs9, %r669;
	cvt.rn.bf16.f32 	%rs10, %r670;
	cvt.rn.bf16.f32 	%rs13, %r671;
	cvt.rn.bf16.f32 	%rs14, %r672;
	cvt.rn.bf16.f32 	%rs17, %r673;
	cvt.rn.bf16.f32 	%rs18, %r674;
	cvt.rn.bf16.f32 	%rs21, %r675;
	cvt.rn.bf16.f32 	%rs22, %r676;
	cvt.rn.bf16.f32 	%rs25, %r677;
	cvt.rn.bf16.f32 	%rs26, %r678;
	cvt.rn.bf16.f32 	%rs29, %r679;
	cvt.rn.bf16.f32 	%rs30, %r680;
	cvt.rn.bf16.f32 	%rs3, %r681;
	cvt.rn.bf16.f32 	%rs4, %r682;
	cvt.rn.bf16.f32 	%rs7, %r683;
	cvt.rn.bf16.f32 	%rs8, %r684;
	cvt.rn.bf16.f32 	%rs11, %r685;
	cvt.rn.bf16.f32 	%rs12, %r686;
	cvt.rn.bf16.f32 	%rs15, %r687;
	cvt.rn.bf16.f32 	%rs16, %r688;
	cvt.rn.bf16.f32 	%rs19, %r689;
	cvt.rn.bf16.f32 	%rs20, %r690;
	cvt.rn.bf16.f32 	%rs23, %r691;
	cvt.rn.bf16.f32 	%rs24, %r692;
	cvt.rn.bf16.f32 	%rs27, %r693;
	cvt.rn.bf16.f32 	%rs28, %r694;
	cvt.rn.bf16.f32 	%rs31, %r695;
	cvt.rn.bf16.f32 	%rs32, %r696;
	.loc	1 103 8                         // sk11_vision.py:103:8
	shl.b32 	%r701, %r219, 12;
	shl.b32 	%r702, %r2, 8;
	and.b32 	%r703, %r702, 3072;
	shl.b32 	%r704, %r2, 3;
	and.b32 	%r705, %r704, 248;
	shl.b32 	%r706, %r2, 2;
	and.b32 	%r707, %r706, 512;
	or.b32 	%r708, %r701, %r707;
	or.b32 	%r709, %r703, %r705;
	xor.b32 	%r710, %r709, %r4;
	or.b32 	%r711, %r710, %r708;
	add.s32 	%r195, %r96, %r711;
	// begin inline asm
	st.shared.v4.b16 [ %r195 + 0 ], { %rs1, %rs2, %rs3, %rs4 };
	// end inline asm
	add.s32 	%r196, %r195, 256;
	// begin inline asm
	st.shared.v4.b16 [ %r196 + 0 ], { %rs5, %rs6, %rs7, %rs8 };
	// end inline asm
	xor.b32 	%r712, %r711, 8;
	add.s32 	%r197, %r96, %r712;
	// begin inline asm
	st.shared.v4.b16 [ %r197 + 0 ], { %rs9, %rs10, %rs11, %rs12 };
	// end inline asm
	add.s32 	%r198, %r197, 256;
	// begin inline asm
	st.shared.v4.b16 [ %r198 + 0 ], { %rs13, %rs14, %rs15, %rs16 };
	// end inline asm
	xor.b32 	%r713, %r711, 16;
	add.s32 	%r199, %r96, %r713;
	// begin inline asm
	st.shared.v4.b16 [ %r199 + 0 ], { %rs17, %rs18, %rs19, %rs20 };
	// end inline asm
	add.s32 	%r200, %r199, 256;
	// begin inline asm
	st.shared.v4.b16 [ %r200 + 0 ], { %rs21, %rs22, %rs23, %rs24 };
	// end inline asm
	xor.b32 	%r714, %r711, 24;
	add.s32 	%r201, %r96, %r714;
	// begin inline asm
	st.shared.v4.b16 [ %r201 + 0 ], { %rs25, %rs26, %rs27, %rs28 };
	// end inline asm
	add.s32 	%r202, %r201, 256;
	// begin inline asm
	st.shared.v4.b16 [ %r202 + 0 ], { %rs29, %rs30, %rs31, %rs32 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r715, %r2, 6;
	and.b32 	%r716, %r715, 3072;
	shl.b32 	%r717, %r219, 5;
	and.b32 	%r718, %r9, 504;
	or.b32 	%r719, %r716, %r717;
	xor.b32 	%r720, %r719, %r718;
	add.s32 	%r721, %r96, %r720;
	ld.shared.v2.b32 	{%r203, %r211}, [%r721];
	ld.shared.v2.b32 	{%r207, %r215}, [%r721+512];
	xor.b32 	%r722, %r720, 8;
	add.s32 	%r723, %r96, %r722;
	ld.shared.v2.b32 	{%r204, %r212}, [%r723+4096];
	ld.shared.v2.b32 	{%r208, %r216}, [%r723+4608];
	xor.b32 	%r724, %r720, 16;
	add.s32 	%r725, %r96, %r724;
	ld.shared.v2.b32 	{%r205, %r213}, [%r725+8192];
	ld.shared.v2.b32 	{%r209, %r217}, [%r725+8704];
	xor.b32 	%r726, %r720, 24;
	add.s32 	%r727, %r96, %r726;
	ld.shared.v2.b32 	{%r206, %r214}, [%r727+12288];
	ld.shared.v2.b32 	{%r210, %r218}, [%r727+12800];
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd56 + 0 ], { %r203, %r204, %r205, %r206 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd57 + 0 ], { %r207, %r208, %r209, %r210 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd58 + 0 ], { %r211, %r212, %r213, %r214 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd59 + 0 ], { %r215, %r216, %r217, %r218 };
	// end inline asm
	.loc	1 101 4                         // sk11_vision.py:101:4
	ret;
$L__tmp6:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk11_vision.py"
	.file	2 "/usr/local/lib/python3.12/dist-packages/triton/language/standard.py"
	.section	.debug_abbrev
	{
.b8 1                                   // Abbreviation Code
.b8 17                                  // DW_TAG_compile_unit
.b8 1                                   // DW_CHILDREN_yes
.b8 37                                  // DW_AT_producer
.b8 8                                   // DW_FORM_string
.b8 19                                  // DW_AT_language
.b8 5                                   // DW_FORM_data2
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 16                                  // DW_AT_stmt_list
.b8 6                                   // DW_FORM_data4
.b8 27                                  // DW_AT_comp_dir
.b8 8                                   // DW_FORM_string
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 2                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 0                                   // DW_CHILDREN_no
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 32                                  // DW_AT_inline
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 3                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 1                                   // DW_CHILDREN_yes
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 4                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 0                                   // DW_CHILDREN_no
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 11                                  // DW_FORM_data1
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 186                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xb3 DW_TAG_compile_unit
.b8 116                                 // DW_AT_producer
.b8 114
.b8 105
.b8 116
.b8 111
.b8 110
.b8 0
.b8 2                                   // DW_AT_language
.b8 0
.b8 115                                 // DW_AT_name
.b8 107
.b8 49
.b8 49
.b8 95
.b8 118
.b8 105
.b8 115
.b8 105
.b8 111
.b8 110
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 114
.b8 101
.b8 112
.b8 111
.b8 47
.b8 118
.b8 108
.b8 108
.b8 109
.b8 47
.b8 95
.b8 103
.b8 101
.b8 110
.b8 101
.b8 115
.b8 105
.b8 115
.b8 47
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 115
.b8 0
.b8 2                                   // Abbrev [2] 0x44:0x1b DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 49
.b8 49
.b8 95
.b8 108
.b8 105
.b8 110
.b8 101
.b8 97
.b8 114
.b8 95
.b8 103
.b8 101
.b8 108
.b8 117
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x5f:0x5e DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 68                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x74:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 74                                  // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x8c:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 75                                  // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0xa4:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp4                           // DW_AT_low_pc
.b64 $L__tmp5                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 90                                  // DW_AT_call_line
.b8 33                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_1 = _Nativo(
    "sk11_vision/tile64x128x64_shift0_abi10",
    _PTX_1, "_sk11_linear_gelu_kernel",
    warps=8, shared=73728,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 10, 11],
    horneado={8: 1, 9: 1, 12: 1, 13: 1, 14: 64, 15: 128, 16: 64, 17: 8},
    div16=[5, 6, 7, 10, 11],
)

_PTX_2 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk11_linear_gelu_kernel // -- Begin function _sk11_linear_gelu_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk11_linear_gelu_kernel
.visible .entry _sk11_linear_gelu_kernel(
	.param .u64 .ptr .global .align 1 _sk11_linear_gelu_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk11_linear_gelu_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk11_linear_gelu_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk11_linear_gelu_kernel_param_3,
	.param .u32 _sk11_linear_gelu_kernel_param_4,
	.param .u32 _sk11_linear_gelu_kernel_param_5,
	.param .u32 _sk11_linear_gelu_kernel_param_6,
	.param .u32 _sk11_linear_gelu_kernel_param_7,
	.param .u32 _sk11_linear_gelu_kernel_param_8,
	.param .u32 _sk11_linear_gelu_kernel_param_9,
	.param .u64 .ptr .global .align 1 _sk11_linear_gelu_kernel_param_10,
	.param .u64 .ptr .global .align 1 _sk11_linear_gelu_kernel_param_11
)
.reqntid 256
{
	.reg .pred 	%p<40>;
	.reg .b16 	%rs<18>;
	.reg .b32 	%r<2432>;
	.reg .b64 	%rd<130>;
	.loc	1 66 0                          // sk11_vision.py:66:0
$L__func_begin0:
	.loc	1 66 0                          // sk11_vision.py:66:0

// %bb.0:
	ld.param.b32 	%r17, [_sk11_linear_gelu_kernel_param_9];
	ld.param.b32 	%r16, [_sk11_linear_gelu_kernel_param_5];
	ld.param.b32 	%r15, [_sk11_linear_gelu_kernel_param_4];
	ld.param.b64 	%rd17, [_sk11_linear_gelu_kernel_param_3];
	ld.param.b64 	%rd16, [_sk11_linear_gelu_kernel_param_2];
	ld.param.b64 	%rd15, [_sk11_linear_gelu_kernel_param_1];
	ld.param.b64 	%rd14, [_sk11_linear_gelu_kernel_param_0];
$L__tmp0:
	.loc	1 73 24                         // sk11_vision.py:73:24
	mov.u32 	%r44, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk11_vision.py:74:27 ]
	add.s32 	%r45, %r15, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk11_vision.py:74:27 ]
	shr.s32 	%r46, %r45, 31;
	shr.u32 	%r47, %r46, 25;
	add.s32 	%r48, %r45, %r47;
	shr.s32 	%r49, %r48, 7;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk11_vision.py:75:27 ]
	add.s32 	%r50, %r16, 255;
	.loc	2 43 30                         // standard.py:43:30 @[ sk11_vision.py:75:27 ]
	shr.s32 	%r51, %r50, 31;
	shr.u32 	%r52, %r51, 24;
	add.s32 	%r53, %r50, %r52;
	shr.s32 	%r54, %r53, 8;
$L__tmp3:
	.loc	1 76 29                         // sk11_vision.py:76:29
	shl.b32 	%r55, %r54, 3;
	ld.param.b32 	%r56, [_sk11_linear_gelu_kernel_param_6];
	ld.param.b32 	%r57, [_sk11_linear_gelu_kernel_param_7];
	.loc	1 77 22                         // sk11_vision.py:77:22
	div.s32 	%r58, %r44, %r55;
	ld.param.b32 	%r59, [_sk11_linear_gelu_kernel_param_8];
	.loc	1 77 38                         // sk11_vision.py:77:38
	shl.b32 	%r60, %r58, 3;
	.loc	1 78 30                         // sk11_vision.py:78:30
	sub.s32 	%r61, %r49, %r60;
	.loc	1 78 39                         // sk11_vision.py:78:39
	min.s32 	%r62, %r61, 8;
	.loc	1 79 30                         // sk11_vision.py:79:30
	mul.lo.s32 	%r63, %r58, %r55;
	sub.s32 	%r64, %r44, %r63;
	.loc	1 80 36                         // sk11_vision.py:80:36
	div.s32 	%r65, %r64, %r62;
	.loc	1 79 46                         // sk11_vision.py:79:46
	mul.lo.s32 	%r66, %r65, %r62;
	sub.s32 	%r67, %r64, %r66;
	.loc	1 79 23                         // sk11_vision.py:79:23
	add.s32 	%r68, %r67, %r60;
	.loc	1 82 22                         // sk11_vision.py:82:22
	shl.b32 	%r1, %r68, 7;
	.loc	1 82 45                         // sk11_vision.py:82:45
	mov.u32 	%r2, %tid.x;
	shr.u32 	%r69, %r2, 3;
	bfe.u32 	%r70, %r2, 3, 5;
	or.b32 	%r71, %r70, 32;
	or.b32 	%r72, %r70, 64;
	or.b32 	%r73, %r69, 96;
	.loc	1 82 32                         // sk11_vision.py:82:32
	or.b32 	%r74, %r1, %r70;
	or.b32 	%r75, %r1, %r71;
	or.b32 	%r76, %r1, %r72;
	or.b32 	%r77, %r1, %r73;
	.loc	1 82 57                         // sk11_vision.py:82:57
	rem.s32 	%r78, %r74, %r15;
	rem.s32 	%r79, %r75, %r15;
	rem.s32 	%r80, %r76, %r15;
	rem.s32 	%r81, %r77, %r15;
	.loc	1 83 22                         // sk11_vision.py:83:22
	shl.b32 	%r3, %r65, 8;
	.loc	1 83 45                         // sk11_vision.py:83:45
	and.b32 	%r4, %r2, 255;
	.loc	1 83 32                         // sk11_vision.py:83:32
	or.b32 	%r82, %r3, %r70;
	or.b32 	%r83, %r3, %r71;
	or.b32 	%r84, %r3, %r72;
	or.b32 	%r85, %r3, %r73;
	or.b32 	%r86, %r82, 128;
	or.b32 	%r87, %r82, 160;
	or.b32 	%r88, %r82, 192;
	or.b32 	%r89, %r69, %r3;
	or.b32 	%r90, %r89, 224;
	.loc	1 83 57                         // sk11_vision.py:83:57
	rem.s32 	%r91, %r82, %r16;
	rem.s32 	%r92, %r83, %r16;
	rem.s32 	%r93, %r84, %r16;
	rem.s32 	%r94, %r85, %r16;
	rem.s32 	%r95, %r86, %r16;
	rem.s32 	%r96, %r87, %r16;
	rem.s32 	%r97, %r88, %r16;
	rem.s32 	%r98, %r90, %r16;
	.loc	1 86 39                         // sk11_vision.py:86:39
	mul.lo.s32 	%r99, %r78, %r57;
	mul.lo.s32 	%r100, %r79, %r57;
	mul.lo.s32 	%r101, %r80, %r57;
	mul.lo.s32 	%r102, %r81, %r57;
	.loc	1 86 21                         // sk11_vision.py:86:21
	mad.wide.s32 	%rd42, %r99, 2, %rd14;
	mad.wide.s32 	%rd43, %r100, 2, %rd14;
	mad.wide.s32 	%rd44, %r101, 2, %rd14;
	mad.wide.s32 	%rd45, %r102, 2, %rd14;
	.loc	1 86 58                         // sk11_vision.py:86:58
	and.b32 	%r5, %r2, 7;
	shl.b32 	%r103, %r5, 3;
	.loc	1 86 51                         // sk11_vision.py:86:51
	mul.wide.u32 	%rd46, %r103, 2;
	add.s64 	%rd18, %rd42, %rd46;
	add.s64 	%rd19, %rd43, %rd46;
	add.s64 	%rd20, %rd44, %rd46;
	add.s64 	%rd21, %rd45, %rd46;
	.loc	1 87 21                         // sk11_vision.py:87:21
	add.s64 	%rd47, %rd15, %rd46;
	.loc	1 87 69                         // sk11_vision.py:87:69
	mul.lo.s32 	%r104, %r91, %r59;
	mul.lo.s32 	%r105, %r92, %r59;
	mul.lo.s32 	%r106, %r93, %r59;
	mul.lo.s32 	%r107, %r94, %r59;
	mul.lo.s32 	%r108, %r95, %r59;
	mul.lo.s32 	%r109, %r96, %r59;
	mul.lo.s32 	%r110, %r97, %r59;
	mul.lo.s32 	%r111, %r98, %r59;
	.loc	1 87 51                         // sk11_vision.py:87:51
	mad.wide.s32 	%rd22, %r104, 2, %rd47;
	mad.wide.s32 	%rd23, %r105, 2, %rd47;
	mad.wide.s32 	%rd24, %r106, 2, %rd47;
	mad.wide.s32 	%rd25, %r107, 2, %rd47;
	mad.wide.s32 	%rd26, %r108, 2, %rd47;
	mad.wide.s32 	%rd27, %r109, 2, %rd47;
	mad.wide.s32 	%rd28, %r110, 2, %rd47;
	mad.wide.s32 	%rd29, %r111, 2, %rd47;
$L__tmp4:
	.loc	2 43 17                         // standard.py:43:17 @[ sk11_vision.py:90:33 ]
	add.s32 	%r112, %r56, 63;
$L__tmp5:
	.loc	1 90 22                         // sk11_vision.py:90:22
	setp.gt.s32 	%p1, %r112, 63;
	.loc	1 91 30                         // sk11_vision.py:91:30
	shl.b32 	%r116, %r4, 4;
	shl.b32 	%r7, %r2, 1;
	and.b32 	%r117, %r7, 112;
	xor.b32 	%r118, %r116, %r117;
	mov.b32 	%r119, global_smem;
	add.s32 	%r23, %r119, %r118;
	add.s32 	%r18, %r23, 65536;
	selp.b32 	%r19, 16, 0, %p1;
	// begin inline asm
	cp.async.cg.shared.global [ %r18 + 0 ], [ %rd18 + 0 ], 0x10, %r19;
	// end inline asm
	add.s32 	%r20, %r23, 69632;
	// begin inline asm
	cp.async.cg.shared.global [ %r20 + 0 ], [ %rd19 + 0 ], 0x10, %r19;
	// end inline asm
	add.s32 	%r21, %r23, 73728;
	// begin inline asm
	cp.async.cg.shared.global [ %r21 + 0 ], [ %rd20 + 0 ], 0x10, %r19;
	// end inline asm
	add.s32 	%r22, %r23, 77824;
	// begin inline asm
	cp.async.cg.shared.global [ %r22 + 0 ], [ %rd21 + 0 ], 0x10, %r19;
	// end inline asm
	cp.async.commit_group;
	.loc	1 91 47                         // sk11_vision.py:91:47
	// begin inline asm
	cp.async.cg.shared.global [ %r23 + 0 ], [ %rd22 + 0 ], 0x10, %r19;
	// end inline asm
	add.s32 	%r24, %r23, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r24 + 0 ], [ %rd23 + 0 ], 0x10, %r19;
	// end inline asm
	add.s32 	%r25, %r23, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r25 + 0 ], [ %rd24 + 0 ], 0x10, %r19;
	// end inline asm
	add.s32 	%r26, %r23, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r26 + 0 ], [ %rd25 + 0 ], 0x10, %r19;
	// end inline asm
	add.s32 	%r27, %r23, 16384;
	// begin inline asm
	cp.async.cg.shared.global [ %r27 + 0 ], [ %rd26 + 0 ], 0x10, %r19;
	// end inline asm
	add.s32 	%r28, %r23, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r28 + 0 ], [ %rd27 + 0 ], 0x10, %r19;
	// end inline asm
	add.s32 	%r29, %r23, 24576;
	// begin inline asm
	cp.async.cg.shared.global [ %r29 + 0 ], [ %rd28 + 0 ], 0x10, %r19;
	// end inline asm
	add.s32 	%r30, %r23, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r30 + 0 ], [ %rd29 + 0 ], 0x10, %r19;
	// end inline asm
	cp.async.commit_group;
	.loc	1 90 22                         // sk11_vision.py:90:22
	setp.gt.s32 	%p2, %r112, 127;
	.loc	1 92 18                         // sk11_vision.py:92:18
	add.s64 	%rd30, %rd18, 128;
	add.s64 	%rd31, %rd19, 128;
	add.s64 	%rd32, %rd20, 128;
	add.s64 	%rd33, %rd21, 128;
	.loc	1 93 18                         // sk11_vision.py:93:18
	add.s64 	%rd34, %rd22, 128;
	add.s64 	%rd35, %rd23, 128;
	add.s64 	%rd36, %rd24, 128;
	add.s64 	%rd37, %rd25, 128;
	add.s64 	%rd38, %rd26, 128;
	add.s64 	%rd39, %rd27, 128;
	add.s64 	%rd40, %rd28, 128;
	add.s64 	%rd41, %rd29, 128;
	.loc	1 91 30                         // sk11_vision.py:91:30
	bar.sync 	0;
	add.s32 	%r31, %r23, 81920;
	selp.b32 	%r32, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r31 + 0 ], [ %rd30 + 0 ], 0x10, %r32;
	// end inline asm
	add.s32 	%r33, %r23, 86016;
	// begin inline asm
	cp.async.cg.shared.global [ %r33 + 0 ], [ %rd31 + 0 ], 0x10, %r32;
	// end inline asm
	add.s32 	%r34, %r23, 90112;
	// begin inline asm
	cp.async.cg.shared.global [ %r34 + 0 ], [ %rd32 + 0 ], 0x10, %r32;
	// end inline asm
	add.s32 	%r35, %r23, 94208;
	// begin inline asm
	cp.async.cg.shared.global [ %r35 + 0 ], [ %rd33 + 0 ], 0x10, %r32;
	// end inline asm
	cp.async.commit_group;
	.loc	1 91 47                         // sk11_vision.py:91:47
	add.s32 	%r36, %r23, 32768;
	// begin inline asm
	cp.async.cg.shared.global [ %r36 + 0 ], [ %rd34 + 0 ], 0x10, %r32;
	// end inline asm
	add.s32 	%r37, %r23, 36864;
	// begin inline asm
	cp.async.cg.shared.global [ %r37 + 0 ], [ %rd35 + 0 ], 0x10, %r32;
	// end inline asm
	add.s32 	%r38, %r23, 40960;
	// begin inline asm
	cp.async.cg.shared.global [ %r38 + 0 ], [ %rd36 + 0 ], 0x10, %r32;
	// end inline asm
	add.s32 	%r39, %r23, 45056;
	// begin inline asm
	cp.async.cg.shared.global [ %r39 + 0 ], [ %rd37 + 0 ], 0x10, %r32;
	// end inline asm
	add.s32 	%r40, %r23, 49152;
	// begin inline asm
	cp.async.cg.shared.global [ %r40 + 0 ], [ %rd38 + 0 ], 0x10, %r32;
	// end inline asm
	add.s32 	%r41, %r23, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r41 + 0 ], [ %rd39 + 0 ], 0x10, %r32;
	// end inline asm
	add.s32 	%r42, %r23, 57344;
	// begin inline asm
	cp.async.cg.shared.global [ %r42 + 0 ], [ %rd40 + 0 ], 0x10, %r32;
	// end inline asm
	add.s32 	%r43, %r23, 61440;
	// begin inline asm
	cp.async.cg.shared.global [ %r43 + 0 ], [ %rd41 + 0 ], 0x10, %r32;
	// end inline asm
	cp.async.commit_group;
	.loc	1 90 22                         // sk11_vision.py:90:22
	@%p1 bra 	$L__BB0_2;
	bra.uni 	$L__BB0_1;
$L__BB0_2:                              // %.lr.ph
	.loc	1 0 22                          // sk11_vision.py:0:22
	cvt.s64.s32 	%rd1, %r99;
	cvt.s64.s32 	%rd2, %r100;
	cvt.s64.s32 	%rd3, %r101;
	cvt.s64.s32 	%rd4, %r102;
	cvt.s64.s32 	%rd5, %r104;
	cvt.s64.s32 	%rd6, %r105;
	cvt.s64.s32 	%rd7, %r106;
	cvt.s64.s32 	%rd8, %r107;
	cvt.s64.s32 	%rd9, %r108;
	cvt.s64.s32 	%rd10, %r109;
	cvt.s64.s32 	%rd11, %r110;
	cvt.s64.s32 	%rd12, %r111;
	shr.s32 	%r113, %r112, 31;
	shr.u32 	%r114, %r113, 26;
	add.s32 	%r115, %r112, %r114;
	shr.s32 	%r6, %r115, 6;
	add.s32 	%r8, %r6, -2;
	shl.b32 	%r120, %r2, 7;
	and.b32 	%r121, %r120, 1920;
	shl.b32 	%r122, %r2, 4;
	and.b32 	%r123, %r122, 2160;
	and.b32 	%r124, %r2, 16;
	or.b32 	%r125, %r121, %r123;
	xor.b32 	%r9, %r125, %r124;
	xor.b32 	%r10, %r9, 32;
	xor.b32 	%r11, %r9, 64;
	xor.b32 	%r12, %r9, 96;
	shl.b32 	%r126, %r5, 7;
	shl.b32 	%r127, %r2, 5;
	and.b32 	%r128, %r127, 3072;
	shl.b32 	%r2431, %r5, 4;
	and.b32 	%r129, %r7, 48;
	or.b32 	%r130, %r126, %r128;
	xor.b32 	%r131, %r2431, %r129;
	or.b32 	%r13, %r130, %r131;
	xor.b32 	%r14, %r13, 64;
	.loc	1 90 22                         // sk11_vision.py:90:22
	mul.wide.u32 	%rd13, %r5, 16;
	shl.b64 	%rd48, %rd12, 1;
	add.s64 	%rd49, %rd48, %rd15;
	add.s64 	%rd129, %rd49, 256;
	shl.b64 	%rd50, %rd11, 1;
	add.s64 	%rd51, %rd50, %rd15;
	add.s64 	%rd128, %rd51, 256;
	shl.b64 	%rd52, %rd10, 1;
	add.s64 	%rd53, %rd52, %rd15;
	add.s64 	%rd127, %rd53, 256;
	shl.b64 	%rd54, %rd9, 1;
	add.s64 	%rd55, %rd54, %rd15;
	add.s64 	%rd126, %rd55, 256;
	shl.b64 	%rd56, %rd8, 1;
	add.s64 	%rd57, %rd56, %rd15;
	add.s64 	%rd125, %rd57, 256;
	shl.b64 	%rd58, %rd7, 1;
	add.s64 	%rd59, %rd58, %rd15;
	add.s64 	%rd124, %rd59, 256;
	shl.b64 	%rd60, %rd6, 1;
	add.s64 	%rd61, %rd60, %rd15;
	add.s64 	%rd123, %rd61, 256;
	shl.b64 	%rd62, %rd5, 1;
	add.s64 	%rd63, %rd62, %rd15;
	add.s64 	%rd122, %rd63, 256;
	shl.b64 	%rd64, %rd4, 1;
	add.s64 	%rd65, %rd64, %rd14;
	add.s64 	%rd121, %rd65, 256;
	shl.b64 	%rd66, %rd3, 1;
	add.s64 	%rd67, %rd66, %rd14;
	add.s64 	%rd120, %rd67, 256;
	shl.b64 	%rd68, %rd2, 1;
	add.s64 	%rd69, %rd68, %rd14;
	add.s64 	%rd119, %rd69, 256;
	shl.b64 	%rd70, %rd1, 1;
	add.s64 	%rd71, %rd70, %rd14;
	add.s64 	%rd118, %rd71, 256;
	mov.b32 	%r2430, 0;
	mov.b32 	%r2302, 0f00000000;
	mov.b32 	%r2301, 1;
	mov.b32 	%r2300, -1;
	mov.b32 	%r2303, %r2302;
	mov.b32 	%r2304, %r2302;
	mov.b32 	%r2305, %r2302;
	mov.b32 	%r2306, %r2302;
	mov.b32 	%r2307, %r2302;
	mov.b32 	%r2308, %r2302;
	mov.b32 	%r2309, %r2302;
	mov.b32 	%r2310, %r2302;
	mov.b32 	%r2311, %r2302;
	mov.b32 	%r2312, %r2302;
	mov.b32 	%r2313, %r2302;
	mov.b32 	%r2314, %r2302;
	mov.b32 	%r2315, %r2302;
	mov.b32 	%r2316, %r2302;
	mov.b32 	%r2317, %r2302;
	mov.b32 	%r2318, %r2302;
	mov.b32 	%r2319, %r2302;
	mov.b32 	%r2320, %r2302;
	mov.b32 	%r2321, %r2302;
	mov.b32 	%r2322, %r2302;
	mov.b32 	%r2323, %r2302;
	mov.b32 	%r2324, %r2302;
	mov.b32 	%r2325, %r2302;
	mov.b32 	%r2326, %r2302;
	mov.b32 	%r2327, %r2302;
	mov.b32 	%r2328, %r2302;
	mov.b32 	%r2329, %r2302;
	mov.b32 	%r2330, %r2302;
	mov.b32 	%r2331, %r2302;
	mov.b32 	%r2332, %r2302;
	mov.b32 	%r2333, %r2302;
	mov.b32 	%r2334, %r2302;
	mov.b32 	%r2335, %r2302;
	mov.b32 	%r2336, %r2302;
	mov.b32 	%r2337, %r2302;
	mov.b32 	%r2338, %r2302;
	mov.b32 	%r2339, %r2302;
	mov.b32 	%r2340, %r2302;
	mov.b32 	%r2341, %r2302;
	mov.b32 	%r2342, %r2302;
	mov.b32 	%r2343, %r2302;
	mov.b32 	%r2344, %r2302;
	mov.b32 	%r2345, %r2302;
	mov.b32 	%r2346, %r2302;
	mov.b32 	%r2347, %r2302;
	mov.b32 	%r2348, %r2302;
	mov.b32 	%r2349, %r2302;
	mov.b32 	%r2350, %r2302;
	mov.b32 	%r2351, %r2302;
	mov.b32 	%r2352, %r2302;
	mov.b32 	%r2353, %r2302;
	mov.b32 	%r2354, %r2302;
	mov.b32 	%r2355, %r2302;
	mov.b32 	%r2356, %r2302;
	mov.b32 	%r2357, %r2302;
	mov.b32 	%r2358, %r2302;
	mov.b32 	%r2359, %r2302;
	mov.b32 	%r2360, %r2302;
	mov.b32 	%r2361, %r2302;
	mov.b32 	%r2362, %r2302;
	mov.b32 	%r2363, %r2302;
	mov.b32 	%r2364, %r2302;
	mov.b32 	%r2365, %r2302;
	mov.b32 	%r2366, %r2302;
	mov.b32 	%r2367, %r2302;
	mov.b32 	%r2368, %r2302;
	mov.b32 	%r2369, %r2302;
	mov.b32 	%r2370, %r2302;
	mov.b32 	%r2371, %r2302;
	mov.b32 	%r2372, %r2302;
	mov.b32 	%r2373, %r2302;
	mov.b32 	%r2374, %r2302;
	mov.b32 	%r2375, %r2302;
	mov.b32 	%r2376, %r2302;
	mov.b32 	%r2377, %r2302;
	mov.b32 	%r2378, %r2302;
	mov.b32 	%r2379, %r2302;
	mov.b32 	%r2380, %r2302;
	mov.b32 	%r2381, %r2302;
	mov.b32 	%r2382, %r2302;
	mov.b32 	%r2383, %r2302;
	mov.b32 	%r2384, %r2302;
	mov.b32 	%r2385, %r2302;
	mov.b32 	%r2386, %r2302;
	mov.b32 	%r2387, %r2302;
	mov.b32 	%r2388, %r2302;
	mov.b32 	%r2389, %r2302;
	mov.b32 	%r2390, %r2302;
	mov.b32 	%r2391, %r2302;
	mov.b32 	%r2392, %r2302;
	mov.b32 	%r2393, %r2302;
	mov.b32 	%r2394, %r2302;
	mov.b32 	%r2395, %r2302;
	mov.b32 	%r2396, %r2302;
	mov.b32 	%r2397, %r2302;
	mov.b32 	%r2398, %r2302;
	mov.b32 	%r2399, %r2302;
	mov.b32 	%r2400, %r2302;
	mov.b32 	%r2401, %r2302;
	mov.b32 	%r2402, %r2302;
	mov.b32 	%r2403, %r2302;
	mov.b32 	%r2404, %r2302;
	mov.b32 	%r2405, %r2302;
	mov.b32 	%r2406, %r2302;
	mov.b32 	%r2407, %r2302;
	mov.b32 	%r2408, %r2302;
	mov.b32 	%r2409, %r2302;
	mov.b32 	%r2410, %r2302;
	mov.b32 	%r2411, %r2302;
	mov.b32 	%r2412, %r2302;
	mov.b32 	%r2413, %r2302;
	mov.b32 	%r2414, %r2302;
	mov.b32 	%r2415, %r2302;
	mov.b32 	%r2416, %r2302;
	mov.b32 	%r2417, %r2302;
	mov.b32 	%r2418, %r2302;
	mov.b32 	%r2419, %r2302;
	mov.b32 	%r2420, %r2302;
	mov.b32 	%r2421, %r2302;
	mov.b32 	%r2422, %r2302;
	mov.b32 	%r2423, %r2302;
	mov.b32 	%r2424, %r2302;
	mov.b32 	%r2425, %r2302;
	mov.b32 	%r2426, %r2302;
	mov.b32 	%r2427, %r2302;
	mov.b32 	%r2428, %r2302;
	mov.b32 	%r2429, %r2302;
$L__BB0_3:                              // =>This Inner Loop Header: Depth=1
	setp.lt.s32 	%p3, %r2430, %r8;
	add.s32 	%r273, %r2300, 1;
	setp.gt.s32 	%p4, %r273, 1;
	selp.b32 	%r2300, 0, %r273, %p4;
	.loc	1 91 30                         // sk11_vision.py:91:30
	cp.async.wait_group 	2;
	bar.sync 	0;
	shl.b32 	%r274, %r2300, 14;
	add.s32 	%r275, %r119, %r274;
	add.s32 	%r276, %r275, %r9;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r132, %r133, %r134, %r135}, [%r276+65536];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r152, %r153, %r154, %r155}, [%r276+69632];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r156, %r157, %r158, %r159}, [%r276+73728];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r160, %r161, %r162, %r163}, [%r276+77824];
	add.s32 	%r277, %r275, %r10;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r164, %r165, %r166, %r167}, [%r277+65536];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r184, %r185, %r186, %r187}, [%r277+69632];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r188, %r189, %r190, %r191}, [%r277+73728];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r192, %r193, %r194, %r195}, [%r277+77824];
	add.s32 	%r278, %r275, %r11;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r196, %r197, %r198, %r199}, [%r278+65536];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r216, %r217, %r218, %r219}, [%r278+69632];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r220, %r221, %r222, %r223}, [%r278+73728];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r224, %r225, %r226, %r227}, [%r278+77824];
	add.s32 	%r279, %r275, %r12;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r228, %r229, %r230, %r231}, [%r279+65536];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r248, %r249, %r250, %r251}, [%r279+69632];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r252, %r253, %r254, %r255}, [%r279+73728];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r256, %r257, %r258, %r259}, [%r279+77824];
	.loc	1 91 47                         // sk11_vision.py:91:47
	add.s32 	%r280, %r275, %r274;
	add.s32 	%r281, %r280, %r13;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r136, %r137, %r168, %r169}, [%r281];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r138, %r139, %r170, %r171}, [%r281+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r140, %r141, %r172, %r173}, [%r281+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r142, %r143, %r174, %r175}, [%r281+12288];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r144, %r145, %r176, %r177}, [%r281+16384];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r146, %r147, %r178, %r179}, [%r281+20480];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r148, %r149, %r180, %r181}, [%r281+24576];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r150, %r151, %r182, %r183}, [%r281+28672];
	add.s32 	%r282, %r280, %r14;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r200, %r201, %r232, %r233}, [%r282];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r202, %r203, %r234, %r235}, [%r282+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r204, %r205, %r236, %r237}, [%r282+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r206, %r207, %r238, %r239}, [%r282+12288];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r208, %r209, %r240, %r241}, [%r282+16384];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r210, %r211, %r242, %r243}, [%r282+20480];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r212, %r213, %r244, %r245}, [%r282+24576];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r214, %r215, %r246, %r247}, [%r282+28672];
	.loc	1 91 39                         // sk11_vision.py:91:39
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2302, %r2303, %r2304, %r2305 }, { %r132, %r133, %r134, %r135 }, { %r136, %r137 }, { %r2302, %r2303, %r2304, %r2305 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2306, %r2307, %r2308, %r2309 }, { %r132, %r133, %r134, %r135 }, { %r138, %r139 }, { %r2306, %r2307, %r2308, %r2309 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2310, %r2311, %r2312, %r2313 }, { %r132, %r133, %r134, %r135 }, { %r140, %r141 }, { %r2310, %r2311, %r2312, %r2313 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2314, %r2315, %r2316, %r2317 }, { %r132, %r133, %r134, %r135 }, { %r142, %r143 }, { %r2314, %r2315, %r2316, %r2317 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2318, %r2319, %r2320, %r2321 }, { %r132, %r133, %r134, %r135 }, { %r144, %r145 }, { %r2318, %r2319, %r2320, %r2321 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2322, %r2323, %r2324, %r2325 }, { %r132, %r133, %r134, %r135 }, { %r146, %r147 }, { %r2322, %r2323, %r2324, %r2325 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2326, %r2327, %r2328, %r2329 }, { %r132, %r133, %r134, %r135 }, { %r148, %r149 }, { %r2326, %r2327, %r2328, %r2329 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2330, %r2331, %r2332, %r2333 }, { %r132, %r133, %r134, %r135 }, { %r150, %r151 }, { %r2330, %r2331, %r2332, %r2333 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2334, %r2335, %r2336, %r2337 }, { %r152, %r153, %r154, %r155 }, { %r136, %r137 }, { %r2334, %r2335, %r2336, %r2337 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2338, %r2339, %r2340, %r2341 }, { %r152, %r153, %r154, %r155 }, { %r138, %r139 }, { %r2338, %r2339, %r2340, %r2341 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2342, %r2343, %r2344, %r2345 }, { %r152, %r153, %r154, %r155 }, { %r140, %r141 }, { %r2342, %r2343, %r2344, %r2345 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2346, %r2347, %r2348, %r2349 }, { %r152, %r153, %r154, %r155 }, { %r142, %r143 }, { %r2346, %r2347, %r2348, %r2349 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2350, %r2351, %r2352, %r2353 }, { %r152, %r153, %r154, %r155 }, { %r144, %r145 }, { %r2350, %r2351, %r2352, %r2353 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2354, %r2355, %r2356, %r2357 }, { %r152, %r153, %r154, %r155 }, { %r146, %r147 }, { %r2354, %r2355, %r2356, %r2357 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2358, %r2359, %r2360, %r2361 }, { %r152, %r153, %r154, %r155 }, { %r148, %r149 }, { %r2358, %r2359, %r2360, %r2361 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2362, %r2363, %r2364, %r2365 }, { %r152, %r153, %r154, %r155 }, { %r150, %r151 }, { %r2362, %r2363, %r2364, %r2365 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2366, %r2367, %r2368, %r2369 }, { %r156, %r157, %r158, %r159 }, { %r136, %r137 }, { %r2366, %r2367, %r2368, %r2369 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2370, %r2371, %r2372, %r2373 }, { %r156, %r157, %r158, %r159 }, { %r138, %r139 }, { %r2370, %r2371, %r2372, %r2373 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2374, %r2375, %r2376, %r2377 }, { %r156, %r157, %r158, %r159 }, { %r140, %r141 }, { %r2374, %r2375, %r2376, %r2377 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2378, %r2379, %r2380, %r2381 }, { %r156, %r157, %r158, %r159 }, { %r142, %r143 }, { %r2378, %r2379, %r2380, %r2381 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2382, %r2383, %r2384, %r2385 }, { %r156, %r157, %r158, %r159 }, { %r144, %r145 }, { %r2382, %r2383, %r2384, %r2385 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2386, %r2387, %r2388, %r2389 }, { %r156, %r157, %r158, %r159 }, { %r146, %r147 }, { %r2386, %r2387, %r2388, %r2389 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2390, %r2391, %r2392, %r2393 }, { %r156, %r157, %r158, %r159 }, { %r148, %r149 }, { %r2390, %r2391, %r2392, %r2393 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2394, %r2395, %r2396, %r2397 }, { %r156, %r157, %r158, %r159 }, { %r150, %r151 }, { %r2394, %r2395, %r2396, %r2397 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2398, %r2399, %r2400, %r2401 }, { %r160, %r161, %r162, %r163 }, { %r136, %r137 }, { %r2398, %r2399, %r2400, %r2401 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2402, %r2403, %r2404, %r2405 }, { %r160, %r161, %r162, %r163 }, { %r138, %r139 }, { %r2402, %r2403, %r2404, %r2405 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2406, %r2407, %r2408, %r2409 }, { %r160, %r161, %r162, %r163 }, { %r140, %r141 }, { %r2406, %r2407, %r2408, %r2409 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2410, %r2411, %r2412, %r2413 }, { %r160, %r161, %r162, %r163 }, { %r142, %r143 }, { %r2410, %r2411, %r2412, %r2413 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2414, %r2415, %r2416, %r2417 }, { %r160, %r161, %r162, %r163 }, { %r144, %r145 }, { %r2414, %r2415, %r2416, %r2417 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2418, %r2419, %r2420, %r2421 }, { %r160, %r161, %r162, %r163 }, { %r146, %r147 }, { %r2418, %r2419, %r2420, %r2421 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2422, %r2423, %r2424, %r2425 }, { %r160, %r161, %r162, %r163 }, { %r148, %r149 }, { %r2422, %r2423, %r2424, %r2425 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2426, %r2427, %r2428, %r2429 }, { %r160, %r161, %r162, %r163 }, { %r150, %r151 }, { %r2426, %r2427, %r2428, %r2429 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2302, %r2303, %r2304, %r2305 }, { %r164, %r165, %r166, %r167 }, { %r168, %r169 }, { %r2302, %r2303, %r2304, %r2305 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2306, %r2307, %r2308, %r2309 }, { %r164, %r165, %r166, %r167 }, { %r170, %r171 }, { %r2306, %r2307, %r2308, %r2309 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2310, %r2311, %r2312, %r2313 }, { %r164, %r165, %r166, %r167 }, { %r172, %r173 }, { %r2310, %r2311, %r2312, %r2313 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2314, %r2315, %r2316, %r2317 }, { %r164, %r165, %r166, %r167 }, { %r174, %r175 }, { %r2314, %r2315, %r2316, %r2317 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2318, %r2319, %r2320, %r2321 }, { %r164, %r165, %r166, %r167 }, { %r176, %r177 }, { %r2318, %r2319, %r2320, %r2321 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2322, %r2323, %r2324, %r2325 }, { %r164, %r165, %r166, %r167 }, { %r178, %r179 }, { %r2322, %r2323, %r2324, %r2325 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2326, %r2327, %r2328, %r2329 }, { %r164, %r165, %r166, %r167 }, { %r180, %r181 }, { %r2326, %r2327, %r2328, %r2329 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2330, %r2331, %r2332, %r2333 }, { %r164, %r165, %r166, %r167 }, { %r182, %r183 }, { %r2330, %r2331, %r2332, %r2333 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2334, %r2335, %r2336, %r2337 }, { %r184, %r185, %r186, %r187 }, { %r168, %r169 }, { %r2334, %r2335, %r2336, %r2337 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2338, %r2339, %r2340, %r2341 }, { %r184, %r185, %r186, %r187 }, { %r170, %r171 }, { %r2338, %r2339, %r2340, %r2341 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2342, %r2343, %r2344, %r2345 }, { %r184, %r185, %r186, %r187 }, { %r172, %r173 }, { %r2342, %r2343, %r2344, %r2345 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2346, %r2347, %r2348, %r2349 }, { %r184, %r185, %r186, %r187 }, { %r174, %r175 }, { %r2346, %r2347, %r2348, %r2349 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2350, %r2351, %r2352, %r2353 }, { %r184, %r185, %r186, %r187 }, { %r176, %r177 }, { %r2350, %r2351, %r2352, %r2353 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2354, %r2355, %r2356, %r2357 }, { %r184, %r185, %r186, %r187 }, { %r178, %r179 }, { %r2354, %r2355, %r2356, %r2357 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2358, %r2359, %r2360, %r2361 }, { %r184, %r185, %r186, %r187 }, { %r180, %r181 }, { %r2358, %r2359, %r2360, %r2361 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2362, %r2363, %r2364, %r2365 }, { %r184, %r185, %r186, %r187 }, { %r182, %r183 }, { %r2362, %r2363, %r2364, %r2365 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2366, %r2367, %r2368, %r2369 }, { %r188, %r189, %r190, %r191 }, { %r168, %r169 }, { %r2366, %r2367, %r2368, %r2369 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2370, %r2371, %r2372, %r2373 }, { %r188, %r189, %r190, %r191 }, { %r170, %r171 }, { %r2370, %r2371, %r2372, %r2373 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2374, %r2375, %r2376, %r2377 }, { %r188, %r189, %r190, %r191 }, { %r172, %r173 }, { %r2374, %r2375, %r2376, %r2377 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2378, %r2379, %r2380, %r2381 }, { %r188, %r189, %r190, %r191 }, { %r174, %r175 }, { %r2378, %r2379, %r2380, %r2381 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2382, %r2383, %r2384, %r2385 }, { %r188, %r189, %r190, %r191 }, { %r176, %r177 }, { %r2382, %r2383, %r2384, %r2385 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2386, %r2387, %r2388, %r2389 }, { %r188, %r189, %r190, %r191 }, { %r178, %r179 }, { %r2386, %r2387, %r2388, %r2389 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2390, %r2391, %r2392, %r2393 }, { %r188, %r189, %r190, %r191 }, { %r180, %r181 }, { %r2390, %r2391, %r2392, %r2393 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2394, %r2395, %r2396, %r2397 }, { %r188, %r189, %r190, %r191 }, { %r182, %r183 }, { %r2394, %r2395, %r2396, %r2397 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2398, %r2399, %r2400, %r2401 }, { %r192, %r193, %r194, %r195 }, { %r168, %r169 }, { %r2398, %r2399, %r2400, %r2401 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2402, %r2403, %r2404, %r2405 }, { %r192, %r193, %r194, %r195 }, { %r170, %r171 }, { %r2402, %r2403, %r2404, %r2405 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2406, %r2407, %r2408, %r2409 }, { %r192, %r193, %r194, %r195 }, { %r172, %r173 }, { %r2406, %r2407, %r2408, %r2409 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2410, %r2411, %r2412, %r2413 }, { %r192, %r193, %r194, %r195 }, { %r174, %r175 }, { %r2410, %r2411, %r2412, %r2413 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2414, %r2415, %r2416, %r2417 }, { %r192, %r193, %r194, %r195 }, { %r176, %r177 }, { %r2414, %r2415, %r2416, %r2417 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2418, %r2419, %r2420, %r2421 }, { %r192, %r193, %r194, %r195 }, { %r178, %r179 }, { %r2418, %r2419, %r2420, %r2421 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2422, %r2423, %r2424, %r2425 }, { %r192, %r193, %r194, %r195 }, { %r180, %r181 }, { %r2422, %r2423, %r2424, %r2425 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2426, %r2427, %r2428, %r2429 }, { %r192, %r193, %r194, %r195 }, { %r182, %r183 }, { %r2426, %r2427, %r2428, %r2429 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2302, %r2303, %r2304, %r2305 }, { %r196, %r197, %r198, %r199 }, { %r200, %r201 }, { %r2302, %r2303, %r2304, %r2305 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2306, %r2307, %r2308, %r2309 }, { %r196, %r197, %r198, %r199 }, { %r202, %r203 }, { %r2306, %r2307, %r2308, %r2309 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2310, %r2311, %r2312, %r2313 }, { %r196, %r197, %r198, %r199 }, { %r204, %r205 }, { %r2310, %r2311, %r2312, %r2313 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2314, %r2315, %r2316, %r2317 }, { %r196, %r197, %r198, %r199 }, { %r206, %r207 }, { %r2314, %r2315, %r2316, %r2317 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2318, %r2319, %r2320, %r2321 }, { %r196, %r197, %r198, %r199 }, { %r208, %r209 }, { %r2318, %r2319, %r2320, %r2321 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2322, %r2323, %r2324, %r2325 }, { %r196, %r197, %r198, %r199 }, { %r210, %r211 }, { %r2322, %r2323, %r2324, %r2325 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2326, %r2327, %r2328, %r2329 }, { %r196, %r197, %r198, %r199 }, { %r212, %r213 }, { %r2326, %r2327, %r2328, %r2329 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2330, %r2331, %r2332, %r2333 }, { %r196, %r197, %r198, %r199 }, { %r214, %r215 }, { %r2330, %r2331, %r2332, %r2333 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2334, %r2335, %r2336, %r2337 }, { %r216, %r217, %r218, %r219 }, { %r200, %r201 }, { %r2334, %r2335, %r2336, %r2337 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2338, %r2339, %r2340, %r2341 }, { %r216, %r217, %r218, %r219 }, { %r202, %r203 }, { %r2338, %r2339, %r2340, %r2341 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2342, %r2343, %r2344, %r2345 }, { %r216, %r217, %r218, %r219 }, { %r204, %r205 }, { %r2342, %r2343, %r2344, %r2345 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2346, %r2347, %r2348, %r2349 }, { %r216, %r217, %r218, %r219 }, { %r206, %r207 }, { %r2346, %r2347, %r2348, %r2349 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2350, %r2351, %r2352, %r2353 }, { %r216, %r217, %r218, %r219 }, { %r208, %r209 }, { %r2350, %r2351, %r2352, %r2353 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2354, %r2355, %r2356, %r2357 }, { %r216, %r217, %r218, %r219 }, { %r210, %r211 }, { %r2354, %r2355, %r2356, %r2357 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2358, %r2359, %r2360, %r2361 }, { %r216, %r217, %r218, %r219 }, { %r212, %r213 }, { %r2358, %r2359, %r2360, %r2361 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2362, %r2363, %r2364, %r2365 }, { %r216, %r217, %r218, %r219 }, { %r214, %r215 }, { %r2362, %r2363, %r2364, %r2365 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2366, %r2367, %r2368, %r2369 }, { %r220, %r221, %r222, %r223 }, { %r200, %r201 }, { %r2366, %r2367, %r2368, %r2369 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2370, %r2371, %r2372, %r2373 }, { %r220, %r221, %r222, %r223 }, { %r202, %r203 }, { %r2370, %r2371, %r2372, %r2373 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2374, %r2375, %r2376, %r2377 }, { %r220, %r221, %r222, %r223 }, { %r204, %r205 }, { %r2374, %r2375, %r2376, %r2377 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2378, %r2379, %r2380, %r2381 }, { %r220, %r221, %r222, %r223 }, { %r206, %r207 }, { %r2378, %r2379, %r2380, %r2381 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2382, %r2383, %r2384, %r2385 }, { %r220, %r221, %r222, %r223 }, { %r208, %r209 }, { %r2382, %r2383, %r2384, %r2385 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2386, %r2387, %r2388, %r2389 }, { %r220, %r221, %r222, %r223 }, { %r210, %r211 }, { %r2386, %r2387, %r2388, %r2389 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2390, %r2391, %r2392, %r2393 }, { %r220, %r221, %r222, %r223 }, { %r212, %r213 }, { %r2390, %r2391, %r2392, %r2393 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2394, %r2395, %r2396, %r2397 }, { %r220, %r221, %r222, %r223 }, { %r214, %r215 }, { %r2394, %r2395, %r2396, %r2397 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2398, %r2399, %r2400, %r2401 }, { %r224, %r225, %r226, %r227 }, { %r200, %r201 }, { %r2398, %r2399, %r2400, %r2401 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2402, %r2403, %r2404, %r2405 }, { %r224, %r225, %r226, %r227 }, { %r202, %r203 }, { %r2402, %r2403, %r2404, %r2405 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2406, %r2407, %r2408, %r2409 }, { %r224, %r225, %r226, %r227 }, { %r204, %r205 }, { %r2406, %r2407, %r2408, %r2409 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2410, %r2411, %r2412, %r2413 }, { %r224, %r225, %r226, %r227 }, { %r206, %r207 }, { %r2410, %r2411, %r2412, %r2413 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2414, %r2415, %r2416, %r2417 }, { %r224, %r225, %r226, %r227 }, { %r208, %r209 }, { %r2414, %r2415, %r2416, %r2417 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2418, %r2419, %r2420, %r2421 }, { %r224, %r225, %r226, %r227 }, { %r210, %r211 }, { %r2418, %r2419, %r2420, %r2421 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2422, %r2423, %r2424, %r2425 }, { %r224, %r225, %r226, %r227 }, { %r212, %r213 }, { %r2422, %r2423, %r2424, %r2425 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2426, %r2427, %r2428, %r2429 }, { %r224, %r225, %r226, %r227 }, { %r214, %r215 }, { %r2426, %r2427, %r2428, %r2429 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2302, %r2303, %r2304, %r2305 }, { %r228, %r229, %r230, %r231 }, { %r232, %r233 }, { %r2302, %r2303, %r2304, %r2305 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2306, %r2307, %r2308, %r2309 }, { %r228, %r229, %r230, %r231 }, { %r234, %r235 }, { %r2306, %r2307, %r2308, %r2309 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2310, %r2311, %r2312, %r2313 }, { %r228, %r229, %r230, %r231 }, { %r236, %r237 }, { %r2310, %r2311, %r2312, %r2313 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2314, %r2315, %r2316, %r2317 }, { %r228, %r229, %r230, %r231 }, { %r238, %r239 }, { %r2314, %r2315, %r2316, %r2317 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2318, %r2319, %r2320, %r2321 }, { %r228, %r229, %r230, %r231 }, { %r240, %r241 }, { %r2318, %r2319, %r2320, %r2321 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2322, %r2323, %r2324, %r2325 }, { %r228, %r229, %r230, %r231 }, { %r242, %r243 }, { %r2322, %r2323, %r2324, %r2325 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2326, %r2327, %r2328, %r2329 }, { %r228, %r229, %r230, %r231 }, { %r244, %r245 }, { %r2326, %r2327, %r2328, %r2329 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2330, %r2331, %r2332, %r2333 }, { %r228, %r229, %r230, %r231 }, { %r246, %r247 }, { %r2330, %r2331, %r2332, %r2333 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2334, %r2335, %r2336, %r2337 }, { %r248, %r249, %r250, %r251 }, { %r232, %r233 }, { %r2334, %r2335, %r2336, %r2337 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2338, %r2339, %r2340, %r2341 }, { %r248, %r249, %r250, %r251 }, { %r234, %r235 }, { %r2338, %r2339, %r2340, %r2341 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2342, %r2343, %r2344, %r2345 }, { %r248, %r249, %r250, %r251 }, { %r236, %r237 }, { %r2342, %r2343, %r2344, %r2345 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2346, %r2347, %r2348, %r2349 }, { %r248, %r249, %r250, %r251 }, { %r238, %r239 }, { %r2346, %r2347, %r2348, %r2349 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2350, %r2351, %r2352, %r2353 }, { %r248, %r249, %r250, %r251 }, { %r240, %r241 }, { %r2350, %r2351, %r2352, %r2353 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2354, %r2355, %r2356, %r2357 }, { %r248, %r249, %r250, %r251 }, { %r242, %r243 }, { %r2354, %r2355, %r2356, %r2357 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2358, %r2359, %r2360, %r2361 }, { %r248, %r249, %r250, %r251 }, { %r244, %r245 }, { %r2358, %r2359, %r2360, %r2361 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2362, %r2363, %r2364, %r2365 }, { %r248, %r249, %r250, %r251 }, { %r246, %r247 }, { %r2362, %r2363, %r2364, %r2365 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2366, %r2367, %r2368, %r2369 }, { %r252, %r253, %r254, %r255 }, { %r232, %r233 }, { %r2366, %r2367, %r2368, %r2369 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2370, %r2371, %r2372, %r2373 }, { %r252, %r253, %r254, %r255 }, { %r234, %r235 }, { %r2370, %r2371, %r2372, %r2373 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2374, %r2375, %r2376, %r2377 }, { %r252, %r253, %r254, %r255 }, { %r236, %r237 }, { %r2374, %r2375, %r2376, %r2377 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2378, %r2379, %r2380, %r2381 }, { %r252, %r253, %r254, %r255 }, { %r238, %r239 }, { %r2378, %r2379, %r2380, %r2381 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2382, %r2383, %r2384, %r2385 }, { %r252, %r253, %r254, %r255 }, { %r240, %r241 }, { %r2382, %r2383, %r2384, %r2385 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2386, %r2387, %r2388, %r2389 }, { %r252, %r253, %r254, %r255 }, { %r242, %r243 }, { %r2386, %r2387, %r2388, %r2389 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2390, %r2391, %r2392, %r2393 }, { %r252, %r253, %r254, %r255 }, { %r244, %r245 }, { %r2390, %r2391, %r2392, %r2393 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2394, %r2395, %r2396, %r2397 }, { %r252, %r253, %r254, %r255 }, { %r246, %r247 }, { %r2394, %r2395, %r2396, %r2397 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2398, %r2399, %r2400, %r2401 }, { %r256, %r257, %r258, %r259 }, { %r232, %r233 }, { %r2398, %r2399, %r2400, %r2401 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2402, %r2403, %r2404, %r2405 }, { %r256, %r257, %r258, %r259 }, { %r234, %r235 }, { %r2402, %r2403, %r2404, %r2405 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2406, %r2407, %r2408, %r2409 }, { %r256, %r257, %r258, %r259 }, { %r236, %r237 }, { %r2406, %r2407, %r2408, %r2409 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2410, %r2411, %r2412, %r2413 }, { %r256, %r257, %r258, %r259 }, { %r238, %r239 }, { %r2410, %r2411, %r2412, %r2413 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2414, %r2415, %r2416, %r2417 }, { %r256, %r257, %r258, %r259 }, { %r240, %r241 }, { %r2414, %r2415, %r2416, %r2417 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2418, %r2419, %r2420, %r2421 }, { %r256, %r257, %r258, %r259 }, { %r242, %r243 }, { %r2418, %r2419, %r2420, %r2421 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2422, %r2423, %r2424, %r2425 }, { %r256, %r257, %r258, %r259 }, { %r244, %r245 }, { %r2422, %r2423, %r2424, %r2425 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 { %r2426, %r2427, %r2428, %r2429 }, { %r256, %r257, %r258, %r259 }, { %r246, %r247 }, { %r2426, %r2427, %r2428, %r2429 };
	// end inline asm
	.loc	1 92 18                         // sk11_vision.py:92:18
	add.s64 	%rd72, %rd118, %rd13;
	add.s64 	%rd73, %rd119, %rd13;
	add.s64 	%rd74, %rd120, %rd13;
	.loc	1 93 18                         // sk11_vision.py:93:18
	add.s64 	%rd75, %rd121, %rd13;
	add.s64 	%rd76, %rd122, %rd13;
	add.s64 	%rd77, %rd123, %rd13;
	add.s64 	%rd78, %rd124, %rd13;
	add.s64 	%rd79, %rd125, %rd13;
	add.s64 	%rd80, %rd126, %rd13;
	add.s64 	%rd81, %rd127, %rd13;
	add.s64 	%rd82, %rd128, %rd13;
	.loc	1 90 22                         // sk11_vision.py:90:22
	add.s64 	%rd83, %rd129, %rd13;
	add.s32 	%r283, %r2301, 1;
	setp.gt.s32 	%p5, %r283, 1;
	selp.b32 	%r2301, 0, %r283, %p5;
	.loc	1 91 30                         // sk11_vision.py:91:30
	shl.b32 	%r284, %r2301, 14;
	bar.sync 	0;
	add.s32 	%r285, %r23, %r284;
	add.s32 	%r260, %r285, 65536;
	selp.b32 	%r261, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r260 + 0 ], [ %rd72 + 0 ], 0x10, %r261;
	// end inline asm
	add.s32 	%r262, %r285, 69632;
	// begin inline asm
	cp.async.cg.shared.global [ %r262 + 0 ], [ %rd73 + 0 ], 0x10, %r261;
	// end inline asm
	add.s32 	%r263, %r285, 73728;
	// begin inline asm
	cp.async.cg.shared.global [ %r263 + 0 ], [ %rd74 + 0 ], 0x10, %r261;
	// end inline asm
	add.s32 	%r264, %r285, 77824;
	// begin inline asm
	cp.async.cg.shared.global [ %r264 + 0 ], [ %rd75 + 0 ], 0x10, %r261;
	// end inline asm
	cp.async.commit_group;
	.loc	1 91 47                         // sk11_vision.py:91:47
	add.s32 	%r265, %r285, %r284;
	// begin inline asm
	cp.async.cg.shared.global [ %r265 + 0 ], [ %rd76 + 0 ], 0x10, %r261;
	// end inline asm
	add.s32 	%r266, %r265, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r266 + 0 ], [ %rd77 + 0 ], 0x10, %r261;
	// end inline asm
	add.s32 	%r267, %r265, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r267 + 0 ], [ %rd78 + 0 ], 0x10, %r261;
	// end inline asm
	add.s32 	%r268, %r265, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r268 + 0 ], [ %rd79 + 0 ], 0x10, %r261;
	// end inline asm
	add.s32 	%r269, %r265, 16384;
	// begin inline asm
	cp.async.cg.shared.global [ %r269 + 0 ], [ %rd80 + 0 ], 0x10, %r261;
	// end inline asm
	add.s32 	%r270, %r265, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r270 + 0 ], [ %rd81 + 0 ], 0x10, %r261;
	// end inline asm
	add.s32 	%r271, %r265, 24576;
	// begin inline asm
	cp.async.cg.shared.global [ %r271 + 0 ], [ %rd82 + 0 ], 0x10, %r261;
	// end inline asm
	add.s32 	%r272, %r265, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r272 + 0 ], [ %rd83 + 0 ], 0x10, %r261;
	// end inline asm
	cp.async.commit_group;
	.loc	1 90 22                         // sk11_vision.py:90:22
	add.s32 	%r2430, %r2430, 1;
	add.s64 	%rd129, %rd129, 128;
	add.s64 	%rd128, %rd128, 128;
	add.s64 	%rd127, %rd127, 128;
	add.s64 	%rd126, %rd126, 128;
	add.s64 	%rd125, %rd125, 128;
	add.s64 	%rd124, %rd124, 128;
	add.s64 	%rd123, %rd123, 128;
	add.s64 	%rd122, %rd122, 128;
	add.s64 	%rd121, %rd121, 128;
	add.s64 	%rd120, %rd120, 128;
	add.s64 	%rd119, %rd119, 128;
	add.s64 	%rd118, %rd118, 128;
	setp.ne.b32 	%p6, %r6, %r2430;
	@%p6 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_4;
$L__BB0_1:                              // %.._crit_edge_crit_edge
	.loc	1 103 8                         // sk11_vision.py:103:8
	shl.b32 	%r2431, %r5, 4;
	mov.b32 	%r2326, 0f00000000;
	mov.b32 	%r2330, %r2326;
	mov.b32 	%r2318, %r2326;
	mov.b32 	%r2322, %r2326;
	mov.b32 	%r2310, %r2326;
	mov.b32 	%r2314, %r2326;
	mov.b32 	%r2302, %r2326;
	mov.b32 	%r2306, %r2326;
	mov.b32 	%r2327, %r2326;
	mov.b32 	%r2331, %r2326;
	mov.b32 	%r2319, %r2326;
	mov.b32 	%r2323, %r2326;
	mov.b32 	%r2311, %r2326;
	mov.b32 	%r2315, %r2326;
	mov.b32 	%r2303, %r2326;
	mov.b32 	%r2307, %r2326;
	mov.b32 	%r2328, %r2326;
	mov.b32 	%r2332, %r2326;
	mov.b32 	%r2320, %r2326;
	mov.b32 	%r2324, %r2326;
	mov.b32 	%r2312, %r2326;
	mov.b32 	%r2316, %r2326;
	mov.b32 	%r2304, %r2326;
	mov.b32 	%r2308, %r2326;
	mov.b32 	%r2329, %r2326;
	mov.b32 	%r2333, %r2326;
	mov.b32 	%r2321, %r2326;
	mov.b32 	%r2325, %r2326;
	mov.b32 	%r2313, %r2326;
	mov.b32 	%r2317, %r2326;
	mov.b32 	%r2305, %r2326;
	mov.b32 	%r2309, %r2326;
	mov.b32 	%r2358, %r2326;
	mov.b32 	%r2362, %r2326;
	mov.b32 	%r2350, %r2326;
	mov.b32 	%r2354, %r2326;
	mov.b32 	%r2342, %r2326;
	mov.b32 	%r2346, %r2326;
	mov.b32 	%r2334, %r2326;
	mov.b32 	%r2338, %r2326;
	mov.b32 	%r2359, %r2326;
	mov.b32 	%r2363, %r2326;
	mov.b32 	%r2351, %r2326;
	mov.b32 	%r2355, %r2326;
	mov.b32 	%r2343, %r2326;
	mov.b32 	%r2347, %r2326;
	mov.b32 	%r2335, %r2326;
	mov.b32 	%r2339, %r2326;
	mov.b32 	%r2360, %r2326;
	mov.b32 	%r2364, %r2326;
	mov.b32 	%r2352, %r2326;
	mov.b32 	%r2356, %r2326;
	mov.b32 	%r2344, %r2326;
	mov.b32 	%r2348, %r2326;
	mov.b32 	%r2336, %r2326;
	mov.b32 	%r2340, %r2326;
	mov.b32 	%r2361, %r2326;
	mov.b32 	%r2365, %r2326;
	mov.b32 	%r2353, %r2326;
	mov.b32 	%r2357, %r2326;
	mov.b32 	%r2345, %r2326;
	mov.b32 	%r2349, %r2326;
	mov.b32 	%r2337, %r2326;
	mov.b32 	%r2341, %r2326;
	mov.b32 	%r2390, %r2326;
	mov.b32 	%r2394, %r2326;
	mov.b32 	%r2382, %r2326;
	mov.b32 	%r2386, %r2326;
	mov.b32 	%r2374, %r2326;
	mov.b32 	%r2378, %r2326;
	mov.b32 	%r2366, %r2326;
	mov.b32 	%r2370, %r2326;
	mov.b32 	%r2391, %r2326;
	mov.b32 	%r2395, %r2326;
	mov.b32 	%r2383, %r2326;
	mov.b32 	%r2387, %r2326;
	mov.b32 	%r2375, %r2326;
	mov.b32 	%r2379, %r2326;
	mov.b32 	%r2367, %r2326;
	mov.b32 	%r2371, %r2326;
	mov.b32 	%r2392, %r2326;
	mov.b32 	%r2396, %r2326;
	mov.b32 	%r2384, %r2326;
	mov.b32 	%r2388, %r2326;
	mov.b32 	%r2376, %r2326;
	mov.b32 	%r2380, %r2326;
	mov.b32 	%r2368, %r2326;
	mov.b32 	%r2372, %r2326;
	mov.b32 	%r2393, %r2326;
	mov.b32 	%r2397, %r2326;
	mov.b32 	%r2385, %r2326;
	mov.b32 	%r2389, %r2326;
	mov.b32 	%r2377, %r2326;
	mov.b32 	%r2381, %r2326;
	mov.b32 	%r2369, %r2326;
	mov.b32 	%r2373, %r2326;
	mov.b32 	%r2422, %r2326;
	mov.b32 	%r2426, %r2326;
	mov.b32 	%r2414, %r2326;
	mov.b32 	%r2418, %r2326;
	mov.b32 	%r2406, %r2326;
	mov.b32 	%r2410, %r2326;
	mov.b32 	%r2398, %r2326;
	mov.b32 	%r2402, %r2326;
	mov.b32 	%r2423, %r2326;
	mov.b32 	%r2427, %r2326;
	mov.b32 	%r2415, %r2326;
	mov.b32 	%r2419, %r2326;
	mov.b32 	%r2407, %r2326;
	mov.b32 	%r2411, %r2326;
	mov.b32 	%r2399, %r2326;
	mov.b32 	%r2403, %r2326;
	mov.b32 	%r2424, %r2326;
	mov.b32 	%r2428, %r2326;
	mov.b32 	%r2416, %r2326;
	mov.b32 	%r2420, %r2326;
	mov.b32 	%r2408, %r2326;
	mov.b32 	%r2412, %r2326;
	mov.b32 	%r2400, %r2326;
	mov.b32 	%r2404, %r2326;
	mov.b32 	%r2425, %r2326;
	mov.b32 	%r2429, %r2326;
	mov.b32 	%r2417, %r2326;
	mov.b32 	%r2421, %r2326;
	mov.b32 	%r2409, %r2326;
	mov.b32 	%r2413, %r2326;
	mov.b32 	%r2401, %r2326;
	mov.b32 	%r2405, %r2326;
$L__BB0_4:                              // %._crit_edge
	.loc	1 83 32                         // sk11_vision.py:83:32
	or.b32 	%r417, %r3, %r4;
	.loc	1 83 57                         // sk11_vision.py:83:57
	rem.s32 	%r418, %r417, %r16;
	.loc	1 83 45                         // sk11_vision.py:83:45
	and.b32 	%r419, %r2, 31;
	shl.b32 	%r420, %r419, 3;
	.loc	1 83 32                         // sk11_vision.py:83:32
	or.b32 	%r421, %r3, %r420;
	.loc	1 82 45                         // sk11_vision.py:82:45
	shr.u32 	%r422, %r2, 5;
	.loc	1 82 32                         // sk11_vision.py:82:32
	or.b32 	%r423, %r422, %r1;
	or.b32 	%r424, %r423, 120;
	.loc	1 82 45                         // sk11_vision.py:82:45
	bfe.u32 	%r425, %r2, 5, 3;
	.loc	1 82 32                         // sk11_vision.py:82:32
	or.b32 	%r426, %r425, %r1;
	or.b32 	%r427, %r426, 112;
	or.b32 	%r428, %r426, 104;
	or.b32 	%r429, %r426, 96;
	or.b32 	%r430, %r423, 88;
	or.b32 	%r431, %r426, 80;
	or.b32 	%r432, %r426, 72;
	or.b32 	%r433, %r426, 64;
	or.b32 	%r434, %r423, 56;
	or.b32 	%r435, %r426, 48;
	or.b32 	%r436, %r426, 40;
	or.b32 	%r437, %r426, 32;
	or.b32 	%r438, %r423, 24;
	or.b32 	%r439, %r426, 16;
	or.b32 	%r440, %r426, 8;
	.loc	1 90 22                         // sk11_vision.py:90:22
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 95 33                         // sk11_vision.py:95:33
	mad.wide.s32 	%rd84, %r418, 2, %rd16;
	.loc	1 95 22                         // sk11_vision.py:95:22
	// begin inline asm
	mov.u16 %rs1, 0x0;
	ld.global.b16 { %rs1 }, [ %rd84 + 0 ];
	// end inline asm
	.loc	1 95 14                         // sk11_vision.py:95:14
	shl.b32 	%r441, %r419, 2;
	and.b32 	%r442, %r7, 384;
	bfe.s32 	%r443, %r2, 5, 1;
	and.b32 	%r444, %r2, 32;
	shr.u32 	%r445, %r444, 4;
	add.s32 	%r446, %r119, %r441;
	add.s32 	%r447, %r446, %r445;
	add.s32 	%r286, %r447, %r442;
	// begin inline asm
	st.shared.b16 [ %r286 + 0 ], %rs1;
	// end inline asm
	bar.sync 	0;
	and.b32 	%r448, %r2, 3;
	shl.b32 	%r449, %r448, 3;
	and.b32 	%r450, %r2, 96;
	add.s32 	%r451, %r119, %r449;
	add.s32 	%r452, %r451, %r450;
	.loc	1 104 31                        // sk11_vision.py:104:31
	setp.lt.s32 	%p23, %r426, %r15;
	setp.lt.s32 	%p24, %r440, %r15;
	setp.lt.s32 	%p25, %r439, %r15;
	setp.lt.s32 	%p26, %r438, %r15;
	setp.lt.s32 	%p27, %r437, %r15;
	setp.lt.s32 	%p28, %r436, %r15;
	setp.lt.s32 	%p29, %r435, %r15;
	setp.lt.s32 	%p30, %r434, %r15;
	setp.lt.s32 	%p31, %r433, %r15;
	setp.lt.s32 	%p32, %r432, %r15;
	setp.lt.s32 	%p33, %r431, %r15;
	setp.lt.s32 	%p34, %r430, %r15;
	setp.lt.s32 	%p35, %r429, %r15;
	setp.lt.s32 	%p36, %r428, %r15;
	setp.lt.s32 	%p37, %r427, %r15;
	setp.lt.s32 	%p38, %r424, %r15;
	.loc	1 104 54                        // sk11_vision.py:104:54
	setp.lt.s32 	%p39, %r421, %r16;
	.loc	1 104 37                        // sk11_vision.py:104:37
	and.pred 	%p7, %p23, %p39;
	and.pred 	%p8, %p24, %p39;
	and.pred 	%p9, %p25, %p39;
	and.pred 	%p10, %p26, %p39;
	and.pred 	%p11, %p27, %p39;
	and.pred 	%p12, %p28, %p39;
	and.pred 	%p13, %p29, %p39;
	and.pred 	%p14, %p30, %p39;
	and.pred 	%p15, %p31, %p39;
	and.pred 	%p16, %p32, %p39;
	and.pred 	%p17, %p33, %p39;
	and.pred 	%p18, %p34, %p39;
	and.pred 	%p19, %p35, %p39;
	and.pred 	%p20, %p36, %p39;
	and.pred 	%p21, %p37, %p39;
	and.pred 	%p22, %p38, %p39;
	.loc	1 102 35                        // sk11_vision.py:102:35
	mul.lo.s32 	%r453, %r426, %r17;
	mul.lo.s32 	%r454, %r440, %r17;
	mul.lo.s32 	%r455, %r439, %r17;
	mul.lo.s32 	%r456, %r438, %r17;
	mul.lo.s32 	%r457, %r437, %r17;
	mul.lo.s32 	%r458, %r436, %r17;
	mul.lo.s32 	%r459, %r435, %r17;
	mul.lo.s32 	%r460, %r434, %r17;
	mul.lo.s32 	%r461, %r433, %r17;
	mul.lo.s32 	%r462, %r432, %r17;
	mul.lo.s32 	%r463, %r431, %r17;
	mul.lo.s32 	%r464, %r430, %r17;
	mul.lo.s32 	%r465, %r429, %r17;
	mul.lo.s32 	%r466, %r428, %r17;
	mul.lo.s32 	%r467, %r427, %r17;
	mul.lo.s32 	%r468, %r424, %r17;
	.loc	1 102 18                        // sk11_vision.py:102:18
	mad.wide.s32 	%rd101, %r453, 2, %rd17;
	mad.wide.s32 	%rd102, %r454, 2, %rd17;
	mad.wide.s32 	%rd103, %r455, 2, %rd17;
	mad.wide.s32 	%rd104, %r456, 2, %rd17;
	mad.wide.s32 	%rd105, %r457, 2, %rd17;
	mad.wide.s32 	%rd106, %r458, 2, %rd17;
	mad.wide.s32 	%rd107, %r459, 2, %rd17;
	mad.wide.s32 	%rd108, %r460, 2, %rd17;
	mad.wide.s32 	%rd109, %r461, 2, %rd17;
	mad.wide.s32 	%rd110, %r462, 2, %rd17;
	mad.wide.s32 	%rd111, %r463, 2, %rd17;
	mad.wide.s32 	%rd112, %r464, 2, %rd17;
	mad.wide.s32 	%rd113, %r465, 2, %rd17;
	mad.wide.s32 	%rd114, %r466, 2, %rd17;
	mad.wide.s32 	%rd115, %r467, 2, %rd17;
	mad.wide.s32 	%rd116, %r468, 2, %rd17;
	.loc	1 102 50                        // sk11_vision.py:102:50
	mul.wide.s32 	%rd117, %r421, 2;
	add.s64 	%rd85, %rd101, %rd117;
	add.s64 	%rd86, %rd102, %rd117;
	add.s64 	%rd87, %rd103, %rd117;
	add.s64 	%rd88, %rd104, %rd117;
	add.s64 	%rd89, %rd105, %rd117;
	add.s64 	%rd90, %rd106, %rd117;
	add.s64 	%rd91, %rd107, %rd117;
	add.s64 	%rd92, %rd108, %rd117;
	add.s64 	%rd93, %rd109, %rd117;
	add.s64 	%rd94, %rd110, %rd117;
	add.s64 	%rd95, %rd111, %rd117;
	add.s64 	%rd96, %rd112, %rd117;
	add.s64 	%rd97, %rd113, %rd117;
	add.s64 	%rd98, %rd114, %rd117;
	add.s64 	%rd99, %rd115, %rd117;
	add.s64 	%rd100, %rd116, %rd117;
	.loc	1 95 14                         // sk11_vision.py:95:14
	ld.shared.v4.b16 	{%rs2, %rs3, %rs4, %rs5}, [%r452];
	.loc	1 95 58                         // sk11_vision.py:95:58
	cvt.f32.bf16 	%r469, %rs2;
	cvt.f32.bf16 	%r470, %rs3;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r471, %r2306, %r470;
	add.f32 	%r472, %r2302, %r469;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r473, %r472, 0f3D372713;
	mul.f32 	%r474, %r471, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r475, %r472, %r473;
	mul.f32 	%r476, %r471, %r474;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r477, %r472, %r475, %r472;
	fma.rn.f32 	%r478, %r471, %r476, %r471;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r479, %r477, 0f3F4C422A;
	mul.f32 	%r480, %r478, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r481, %r471, 0f3F000000;
	mul.f32 	%r482, %r472, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r483, %r479, 0fC0000000;
	mul.f32 	%r484, %r480, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r485, %r483, 0f3FB8AA3B;
	ex2.approx.f32 	%r486, %r485;
	mul.f32 	%r487, %r484, 0f3FB8AA3B;
	ex2.approx.f32 	%r488, %r487;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r489, %r486, 0f3F800000;
	add.f32 	%r490, %r488, 0f3F800000;
	mov.b32 	%r491, 0f40000000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r492, %r491, %r489;
	div.full.f32 	%r493, %r491, %r490;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r494, %r492, 0fBF800000;
	add.f32 	%r495, %r493, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r496, %r495, 0f3F800000;
	add.f32 	%r497, %r494, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r498, %r482, %r497;
	mul.f32 	%r499, %r481, %r496;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r288, %r499, %r498;
	.loc	1 95 58                         // sk11_vision.py:95:58
	cvt.f32.bf16 	%r500, %rs4;
	cvt.f32.bf16 	%r501, %rs5;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r502, %r2307, %r501;
	add.f32 	%r503, %r2303, %r500;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r504, %r503, 0f3D372713;
	mul.f32 	%r505, %r502, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r506, %r503, %r504;
	mul.f32 	%r507, %r502, %r505;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r508, %r503, %r506, %r503;
	fma.rn.f32 	%r509, %r502, %r507, %r502;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r510, %r508, 0f3F4C422A;
	mul.f32 	%r511, %r509, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r512, %r502, 0f3F000000;
	mul.f32 	%r513, %r503, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r514, %r510, 0fC0000000;
	mul.f32 	%r515, %r511, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r516, %r514, 0f3FB8AA3B;
	ex2.approx.f32 	%r517, %r516;
	mul.f32 	%r518, %r515, 0f3FB8AA3B;
	ex2.approx.f32 	%r519, %r518;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r520, %r517, 0f3F800000;
	add.f32 	%r521, %r519, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r522, %r491, %r520;
	div.full.f32 	%r523, %r491, %r521;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r524, %r522, 0fBF800000;
	add.f32 	%r525, %r523, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r526, %r525, 0f3F800000;
	add.f32 	%r527, %r524, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r528, %r513, %r527;
	mul.f32 	%r529, %r512, %r526;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r293, %r529, %r528;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r530, %r2308, %r470;
	add.f32 	%r531, %r2304, %r469;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r532, %r531, 0f3D372713;
	mul.f32 	%r533, %r530, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r534, %r531, %r532;
	mul.f32 	%r535, %r530, %r533;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r536, %r531, %r534, %r531;
	fma.rn.f32 	%r537, %r530, %r535, %r530;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r538, %r536, 0f3F4C422A;
	mul.f32 	%r539, %r537, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r540, %r530, 0f3F000000;
	mul.f32 	%r541, %r531, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r542, %r538, 0fC0000000;
	mul.f32 	%r543, %r539, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r544, %r542, 0f3FB8AA3B;
	ex2.approx.f32 	%r545, %r544;
	mul.f32 	%r546, %r543, 0f3FB8AA3B;
	ex2.approx.f32 	%r547, %r546;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r548, %r545, 0f3F800000;
	add.f32 	%r549, %r547, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r550, %r491, %r548;
	div.full.f32 	%r551, %r491, %r549;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r552, %r550, 0fBF800000;
	add.f32 	%r553, %r551, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r554, %r553, 0f3F800000;
	add.f32 	%r555, %r552, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r556, %r541, %r555;
	mul.f32 	%r557, %r540, %r554;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r297, %r557, %r556;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r558, %r2309, %r501;
	add.f32 	%r559, %r2305, %r500;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r560, %r559, 0f3D372713;
	mul.f32 	%r561, %r558, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r562, %r559, %r560;
	mul.f32 	%r563, %r558, %r561;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r564, %r559, %r562, %r559;
	fma.rn.f32 	%r565, %r558, %r563, %r558;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r566, %r564, 0f3F4C422A;
	mul.f32 	%r567, %r565, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r568, %r558, 0f3F000000;
	mul.f32 	%r569, %r559, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r570, %r566, 0fC0000000;
	mul.f32 	%r571, %r567, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r572, %r570, 0f3FB8AA3B;
	ex2.approx.f32 	%r573, %r572;
	mul.f32 	%r574, %r571, 0f3FB8AA3B;
	ex2.approx.f32 	%r575, %r574;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r576, %r573, 0f3F800000;
	add.f32 	%r577, %r575, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r578, %r491, %r576;
	div.full.f32 	%r579, %r491, %r577;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r580, %r578, 0fBF800000;
	add.f32 	%r581, %r579, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r582, %r581, 0f3F800000;
	add.f32 	%r583, %r580, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r584, %r569, %r583;
	mul.f32 	%r585, %r568, %r582;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r301, %r585, %r584;
	.loc	1 95 14                         // sk11_vision.py:95:14
	ld.shared.v4.b16 	{%rs6, %rs7, %rs8, %rs9}, [%r452+128];
	.loc	1 95 58                         // sk11_vision.py:95:58
	cvt.f32.bf16 	%r586, %rs6;
	cvt.f32.bf16 	%r587, %rs7;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r588, %r2314, %r587;
	add.f32 	%r589, %r2310, %r586;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r590, %r589, 0f3D372713;
	mul.f32 	%r591, %r588, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r592, %r589, %r590;
	mul.f32 	%r593, %r588, %r591;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r594, %r589, %r592, %r589;
	fma.rn.f32 	%r595, %r588, %r593, %r588;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r596, %r594, 0f3F4C422A;
	mul.f32 	%r597, %r595, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r598, %r588, 0f3F000000;
	mul.f32 	%r599, %r589, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r600, %r596, 0fC0000000;
	mul.f32 	%r601, %r597, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r602, %r600, 0f3FB8AA3B;
	ex2.approx.f32 	%r603, %r602;
	mul.f32 	%r604, %r601, 0f3FB8AA3B;
	ex2.approx.f32 	%r605, %r604;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r606, %r603, 0f3F800000;
	add.f32 	%r607, %r605, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r608, %r491, %r606;
	div.full.f32 	%r609, %r491, %r607;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r610, %r608, 0fBF800000;
	add.f32 	%r611, %r609, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r612, %r611, 0f3F800000;
	add.f32 	%r613, %r610, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r614, %r599, %r613;
	mul.f32 	%r615, %r598, %r612;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r289, %r615, %r614;
	.loc	1 95 58                         // sk11_vision.py:95:58
	cvt.f32.bf16 	%r616, %rs8;
	cvt.f32.bf16 	%r617, %rs9;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r618, %r2315, %r617;
	add.f32 	%r619, %r2311, %r616;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r620, %r619, 0f3D372713;
	mul.f32 	%r621, %r618, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r622, %r619, %r620;
	mul.f32 	%r623, %r618, %r621;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r624, %r619, %r622, %r619;
	fma.rn.f32 	%r625, %r618, %r623, %r618;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r626, %r624, 0f3F4C422A;
	mul.f32 	%r627, %r625, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r628, %r618, 0f3F000000;
	mul.f32 	%r629, %r619, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r630, %r626, 0fC0000000;
	mul.f32 	%r631, %r627, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r632, %r630, 0f3FB8AA3B;
	ex2.approx.f32 	%r633, %r632;
	mul.f32 	%r634, %r631, 0f3FB8AA3B;
	ex2.approx.f32 	%r635, %r634;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r636, %r633, 0f3F800000;
	add.f32 	%r637, %r635, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r638, %r491, %r636;
	div.full.f32 	%r639, %r491, %r637;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r640, %r638, 0fBF800000;
	add.f32 	%r641, %r639, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r642, %r641, 0f3F800000;
	add.f32 	%r643, %r640, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r644, %r629, %r643;
	mul.f32 	%r645, %r628, %r642;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r294, %r645, %r644;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r646, %r2316, %r587;
	add.f32 	%r647, %r2312, %r586;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r648, %r647, 0f3D372713;
	mul.f32 	%r649, %r646, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r650, %r647, %r648;
	mul.f32 	%r651, %r646, %r649;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r652, %r647, %r650, %r647;
	fma.rn.f32 	%r653, %r646, %r651, %r646;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r654, %r652, 0f3F4C422A;
	mul.f32 	%r655, %r653, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r656, %r646, 0f3F000000;
	mul.f32 	%r657, %r647, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r658, %r654, 0fC0000000;
	mul.f32 	%r659, %r655, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r660, %r658, 0f3FB8AA3B;
	ex2.approx.f32 	%r661, %r660;
	mul.f32 	%r662, %r659, 0f3FB8AA3B;
	ex2.approx.f32 	%r663, %r662;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r664, %r661, 0f3F800000;
	add.f32 	%r665, %r663, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r666, %r491, %r664;
	div.full.f32 	%r667, %r491, %r665;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r668, %r666, 0fBF800000;
	add.f32 	%r669, %r667, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r670, %r669, 0f3F800000;
	add.f32 	%r671, %r668, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r672, %r657, %r671;
	mul.f32 	%r673, %r656, %r670;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r298, %r673, %r672;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r674, %r2317, %r617;
	add.f32 	%r675, %r2313, %r616;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r676, %r675, 0f3D372713;
	mul.f32 	%r677, %r674, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r678, %r675, %r676;
	mul.f32 	%r679, %r674, %r677;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r680, %r675, %r678, %r675;
	fma.rn.f32 	%r681, %r674, %r679, %r674;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r682, %r680, 0f3F4C422A;
	mul.f32 	%r683, %r681, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r684, %r674, 0f3F000000;
	mul.f32 	%r685, %r675, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r686, %r682, 0fC0000000;
	mul.f32 	%r687, %r683, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r688, %r686, 0f3FB8AA3B;
	ex2.approx.f32 	%r689, %r688;
	mul.f32 	%r690, %r687, 0f3FB8AA3B;
	ex2.approx.f32 	%r691, %r690;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r692, %r689, 0f3F800000;
	add.f32 	%r693, %r691, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r694, %r491, %r692;
	div.full.f32 	%r695, %r491, %r693;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r696, %r694, 0fBF800000;
	add.f32 	%r697, %r695, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r698, %r697, 0f3F800000;
	add.f32 	%r699, %r696, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r700, %r685, %r699;
	mul.f32 	%r701, %r684, %r698;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r302, %r701, %r700;
	.loc	1 95 14                         // sk11_vision.py:95:14
	ld.shared.v4.b16 	{%rs10, %rs11, %rs12, %rs13}, [%r452+256];
	.loc	1 95 58                         // sk11_vision.py:95:58
	cvt.f32.bf16 	%r702, %rs10;
	cvt.f32.bf16 	%r703, %rs11;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r704, %r2322, %r703;
	add.f32 	%r705, %r2318, %r702;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r706, %r705, 0f3D372713;
	mul.f32 	%r707, %r704, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r708, %r705, %r706;
	mul.f32 	%r709, %r704, %r707;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r710, %r705, %r708, %r705;
	fma.rn.f32 	%r711, %r704, %r709, %r704;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r712, %r710, 0f3F4C422A;
	mul.f32 	%r713, %r711, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r714, %r704, 0f3F000000;
	mul.f32 	%r715, %r705, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r716, %r712, 0fC0000000;
	mul.f32 	%r717, %r713, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r718, %r716, 0f3FB8AA3B;
	ex2.approx.f32 	%r719, %r718;
	mul.f32 	%r720, %r717, 0f3FB8AA3B;
	ex2.approx.f32 	%r721, %r720;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r722, %r719, 0f3F800000;
	add.f32 	%r723, %r721, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r724, %r491, %r722;
	div.full.f32 	%r725, %r491, %r723;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r726, %r724, 0fBF800000;
	add.f32 	%r727, %r725, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r728, %r727, 0f3F800000;
	add.f32 	%r729, %r726, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r730, %r715, %r729;
	mul.f32 	%r731, %r714, %r728;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r290, %r731, %r730;
	.loc	1 95 58                         // sk11_vision.py:95:58
	cvt.f32.bf16 	%r732, %rs12;
	cvt.f32.bf16 	%r733, %rs13;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r734, %r2323, %r733;
	add.f32 	%r735, %r2319, %r732;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r736, %r735, 0f3D372713;
	mul.f32 	%r737, %r734, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r738, %r735, %r736;
	mul.f32 	%r739, %r734, %r737;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r740, %r735, %r738, %r735;
	fma.rn.f32 	%r741, %r734, %r739, %r734;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r742, %r740, 0f3F4C422A;
	mul.f32 	%r743, %r741, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r744, %r734, 0f3F000000;
	mul.f32 	%r745, %r735, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r746, %r742, 0fC0000000;
	mul.f32 	%r747, %r743, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r748, %r746, 0f3FB8AA3B;
	ex2.approx.f32 	%r749, %r748;
	mul.f32 	%r750, %r747, 0f3FB8AA3B;
	ex2.approx.f32 	%r751, %r750;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r752, %r749, 0f3F800000;
	add.f32 	%r753, %r751, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r754, %r491, %r752;
	div.full.f32 	%r755, %r491, %r753;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r756, %r754, 0fBF800000;
	add.f32 	%r757, %r755, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r758, %r757, 0f3F800000;
	add.f32 	%r759, %r756, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r760, %r745, %r759;
	mul.f32 	%r761, %r744, %r758;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r295, %r761, %r760;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r762, %r2324, %r703;
	add.f32 	%r763, %r2320, %r702;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r764, %r763, 0f3D372713;
	mul.f32 	%r765, %r762, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r766, %r763, %r764;
	mul.f32 	%r767, %r762, %r765;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r768, %r763, %r766, %r763;
	fma.rn.f32 	%r769, %r762, %r767, %r762;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r770, %r768, 0f3F4C422A;
	mul.f32 	%r771, %r769, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r772, %r762, 0f3F000000;
	mul.f32 	%r773, %r763, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r774, %r770, 0fC0000000;
	mul.f32 	%r775, %r771, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r776, %r774, 0f3FB8AA3B;
	ex2.approx.f32 	%r777, %r776;
	mul.f32 	%r778, %r775, 0f3FB8AA3B;
	ex2.approx.f32 	%r779, %r778;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r780, %r777, 0f3F800000;
	add.f32 	%r781, %r779, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r782, %r491, %r780;
	div.full.f32 	%r783, %r491, %r781;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r784, %r782, 0fBF800000;
	add.f32 	%r785, %r783, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r786, %r785, 0f3F800000;
	add.f32 	%r787, %r784, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r788, %r773, %r787;
	mul.f32 	%r789, %r772, %r786;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r299, %r789, %r788;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r790, %r2325, %r733;
	add.f32 	%r791, %r2321, %r732;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r792, %r791, 0f3D372713;
	mul.f32 	%r793, %r790, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r794, %r791, %r792;
	mul.f32 	%r795, %r790, %r793;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r796, %r791, %r794, %r791;
	fma.rn.f32 	%r797, %r790, %r795, %r790;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r798, %r796, 0f3F4C422A;
	mul.f32 	%r799, %r797, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r800, %r790, 0f3F000000;
	mul.f32 	%r801, %r791, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r802, %r798, 0fC0000000;
	mul.f32 	%r803, %r799, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r804, %r802, 0f3FB8AA3B;
	ex2.approx.f32 	%r805, %r804;
	mul.f32 	%r806, %r803, 0f3FB8AA3B;
	ex2.approx.f32 	%r807, %r806;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r808, %r805, 0f3F800000;
	add.f32 	%r809, %r807, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r810, %r491, %r808;
	div.full.f32 	%r811, %r491, %r809;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r812, %r810, 0fBF800000;
	add.f32 	%r813, %r811, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r814, %r813, 0f3F800000;
	add.f32 	%r815, %r812, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r816, %r801, %r815;
	mul.f32 	%r817, %r800, %r814;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r303, %r817, %r816;
	.loc	1 95 14                         // sk11_vision.py:95:14
	ld.shared.v4.b16 	{%rs14, %rs15, %rs16, %rs17}, [%r452+384];
	.loc	1 95 58                         // sk11_vision.py:95:58
	cvt.f32.bf16 	%r818, %rs14;
	cvt.f32.bf16 	%r819, %rs15;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r820, %r2330, %r819;
	add.f32 	%r821, %r2326, %r818;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r822, %r821, 0f3D372713;
	mul.f32 	%r823, %r820, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r824, %r821, %r822;
	mul.f32 	%r825, %r820, %r823;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r826, %r821, %r824, %r821;
	fma.rn.f32 	%r827, %r820, %r825, %r820;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r828, %r826, 0f3F4C422A;
	mul.f32 	%r829, %r827, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r830, %r820, 0f3F000000;
	mul.f32 	%r831, %r821, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r832, %r828, 0fC0000000;
	mul.f32 	%r833, %r829, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r834, %r832, 0f3FB8AA3B;
	ex2.approx.f32 	%r835, %r834;
	mul.f32 	%r836, %r833, 0f3FB8AA3B;
	ex2.approx.f32 	%r837, %r836;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r838, %r835, 0f3F800000;
	add.f32 	%r839, %r837, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r840, %r491, %r838;
	div.full.f32 	%r841, %r491, %r839;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r842, %r840, 0fBF800000;
	add.f32 	%r843, %r841, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r844, %r843, 0f3F800000;
	add.f32 	%r845, %r842, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r846, %r831, %r845;
	mul.f32 	%r847, %r830, %r844;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r291, %r847, %r846;
	.loc	1 95 58                         // sk11_vision.py:95:58
	cvt.f32.bf16 	%r848, %rs16;
	cvt.f32.bf16 	%r849, %rs17;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r850, %r2331, %r849;
	add.f32 	%r851, %r2327, %r848;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r852, %r851, 0f3D372713;
	mul.f32 	%r853, %r850, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r854, %r851, %r852;
	mul.f32 	%r855, %r850, %r853;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r856, %r851, %r854, %r851;
	fma.rn.f32 	%r857, %r850, %r855, %r850;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r858, %r856, 0f3F4C422A;
	mul.f32 	%r859, %r857, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r860, %r850, 0f3F000000;
	mul.f32 	%r861, %r851, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r862, %r858, 0fC0000000;
	mul.f32 	%r863, %r859, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r864, %r862, 0f3FB8AA3B;
	ex2.approx.f32 	%r865, %r864;
	mul.f32 	%r866, %r863, 0f3FB8AA3B;
	ex2.approx.f32 	%r867, %r866;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r868, %r865, 0f3F800000;
	add.f32 	%r869, %r867, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r870, %r491, %r868;
	div.full.f32 	%r871, %r491, %r869;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r872, %r870, 0fBF800000;
	add.f32 	%r873, %r871, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r874, %r873, 0f3F800000;
	add.f32 	%r875, %r872, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r876, %r861, %r875;
	mul.f32 	%r877, %r860, %r874;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r296, %r877, %r876;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r878, %r2332, %r819;
	add.f32 	%r879, %r2328, %r818;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r880, %r879, 0f3D372713;
	mul.f32 	%r881, %r878, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r882, %r879, %r880;
	mul.f32 	%r883, %r878, %r881;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r884, %r879, %r882, %r879;
	fma.rn.f32 	%r885, %r878, %r883, %r878;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r886, %r884, 0f3F4C422A;
	mul.f32 	%r887, %r885, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r888, %r878, 0f3F000000;
	mul.f32 	%r889, %r879, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r890, %r886, 0fC0000000;
	mul.f32 	%r891, %r887, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r892, %r890, 0f3FB8AA3B;
	ex2.approx.f32 	%r893, %r892;
	mul.f32 	%r894, %r891, 0f3FB8AA3B;
	ex2.approx.f32 	%r895, %r894;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r896, %r893, 0f3F800000;
	add.f32 	%r897, %r895, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r898, %r491, %r896;
	div.full.f32 	%r899, %r491, %r897;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r900, %r898, 0fBF800000;
	add.f32 	%r901, %r899, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r902, %r901, 0f3F800000;
	add.f32 	%r903, %r900, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r904, %r889, %r903;
	mul.f32 	%r905, %r888, %r902;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r300, %r905, %r904;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r906, %r2333, %r849;
	add.f32 	%r907, %r2329, %r848;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r908, %r907, 0f3D372713;
	mul.f32 	%r909, %r906, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r910, %r907, %r908;
	mul.f32 	%r911, %r906, %r909;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r912, %r907, %r910, %r907;
	fma.rn.f32 	%r913, %r906, %r911, %r906;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r914, %r912, 0f3F4C422A;
	mul.f32 	%r915, %r913, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r916, %r906, 0f3F000000;
	mul.f32 	%r917, %r907, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r918, %r914, 0fC0000000;
	mul.f32 	%r919, %r915, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r920, %r918, 0f3FB8AA3B;
	ex2.approx.f32 	%r921, %r920;
	mul.f32 	%r922, %r919, 0f3FB8AA3B;
	ex2.approx.f32 	%r923, %r922;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r924, %r921, 0f3F800000;
	add.f32 	%r925, %r923, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r926, %r491, %r924;
	div.full.f32 	%r927, %r491, %r925;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r928, %r926, 0fBF800000;
	add.f32 	%r929, %r927, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r930, %r929, 0f3F800000;
	add.f32 	%r931, %r928, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r932, %r917, %r931;
	mul.f32 	%r933, %r916, %r930;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r304, %r933, %r932;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r934, %r2338, %r470;
	add.f32 	%r935, %r2334, %r469;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r936, %r935, 0f3D372713;
	mul.f32 	%r937, %r934, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r938, %r935, %r936;
	mul.f32 	%r939, %r934, %r937;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r940, %r935, %r938, %r935;
	fma.rn.f32 	%r941, %r934, %r939, %r934;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r942, %r940, 0f3F4C422A;
	mul.f32 	%r943, %r941, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r944, %r934, 0f3F000000;
	mul.f32 	%r945, %r935, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r946, %r942, 0fC0000000;
	mul.f32 	%r947, %r943, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r948, %r946, 0f3FB8AA3B;
	ex2.approx.f32 	%r949, %r948;
	mul.f32 	%r950, %r947, 0f3FB8AA3B;
	ex2.approx.f32 	%r951, %r950;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r952, %r949, 0f3F800000;
	add.f32 	%r953, %r951, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r954, %r491, %r952;
	div.full.f32 	%r955, %r491, %r953;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r956, %r954, 0fBF800000;
	add.f32 	%r957, %r955, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r958, %r957, 0f3F800000;
	add.f32 	%r959, %r956, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r960, %r945, %r959;
	mul.f32 	%r961, %r944, %r958;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r305, %r961, %r960;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r962, %r2339, %r501;
	add.f32 	%r963, %r2335, %r500;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r964, %r963, 0f3D372713;
	mul.f32 	%r965, %r962, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r966, %r963, %r964;
	mul.f32 	%r967, %r962, %r965;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r968, %r963, %r966, %r963;
	fma.rn.f32 	%r969, %r962, %r967, %r962;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r970, %r968, 0f3F4C422A;
	mul.f32 	%r971, %r969, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r972, %r962, 0f3F000000;
	mul.f32 	%r973, %r963, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r974, %r970, 0fC0000000;
	mul.f32 	%r975, %r971, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r976, %r974, 0f3FB8AA3B;
	ex2.approx.f32 	%r977, %r976;
	mul.f32 	%r978, %r975, 0f3FB8AA3B;
	ex2.approx.f32 	%r979, %r978;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r980, %r977, 0f3F800000;
	add.f32 	%r981, %r979, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r982, %r491, %r980;
	div.full.f32 	%r983, %r491, %r981;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r984, %r982, 0fBF800000;
	add.f32 	%r985, %r983, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r986, %r985, 0f3F800000;
	add.f32 	%r987, %r984, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r988, %r973, %r987;
	mul.f32 	%r989, %r972, %r986;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r309, %r989, %r988;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r990, %r2340, %r470;
	add.f32 	%r991, %r2336, %r469;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r992, %r991, 0f3D372713;
	mul.f32 	%r993, %r990, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r994, %r991, %r992;
	mul.f32 	%r995, %r990, %r993;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r996, %r991, %r994, %r991;
	fma.rn.f32 	%r997, %r990, %r995, %r990;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r998, %r996, 0f3F4C422A;
	mul.f32 	%r999, %r997, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1000, %r990, 0f3F000000;
	mul.f32 	%r1001, %r991, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1002, %r998, 0fC0000000;
	mul.f32 	%r1003, %r999, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1004, %r1002, 0f3FB8AA3B;
	ex2.approx.f32 	%r1005, %r1004;
	mul.f32 	%r1006, %r1003, 0f3FB8AA3B;
	ex2.approx.f32 	%r1007, %r1006;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1008, %r1005, 0f3F800000;
	add.f32 	%r1009, %r1007, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1010, %r491, %r1008;
	div.full.f32 	%r1011, %r491, %r1009;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1012, %r1010, 0fBF800000;
	add.f32 	%r1013, %r1011, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1014, %r1013, 0f3F800000;
	add.f32 	%r1015, %r1012, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1016, %r1001, %r1015;
	mul.f32 	%r1017, %r1000, %r1014;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r313, %r1017, %r1016;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1018, %r2341, %r501;
	add.f32 	%r1019, %r2337, %r500;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1020, %r1019, 0f3D372713;
	mul.f32 	%r1021, %r1018, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1022, %r1019, %r1020;
	mul.f32 	%r1023, %r1018, %r1021;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1024, %r1019, %r1022, %r1019;
	fma.rn.f32 	%r1025, %r1018, %r1023, %r1018;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1026, %r1024, 0f3F4C422A;
	mul.f32 	%r1027, %r1025, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1028, %r1018, 0f3F000000;
	mul.f32 	%r1029, %r1019, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1030, %r1026, 0fC0000000;
	mul.f32 	%r1031, %r1027, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1032, %r1030, 0f3FB8AA3B;
	ex2.approx.f32 	%r1033, %r1032;
	mul.f32 	%r1034, %r1031, 0f3FB8AA3B;
	ex2.approx.f32 	%r1035, %r1034;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1036, %r1033, 0f3F800000;
	add.f32 	%r1037, %r1035, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1038, %r491, %r1036;
	div.full.f32 	%r1039, %r491, %r1037;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1040, %r1038, 0fBF800000;
	add.f32 	%r1041, %r1039, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1042, %r1041, 0f3F800000;
	add.f32 	%r1043, %r1040, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1044, %r1029, %r1043;
	mul.f32 	%r1045, %r1028, %r1042;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r317, %r1045, %r1044;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1046, %r2346, %r587;
	add.f32 	%r1047, %r2342, %r586;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1048, %r1047, 0f3D372713;
	mul.f32 	%r1049, %r1046, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1050, %r1047, %r1048;
	mul.f32 	%r1051, %r1046, %r1049;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1052, %r1047, %r1050, %r1047;
	fma.rn.f32 	%r1053, %r1046, %r1051, %r1046;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1054, %r1052, 0f3F4C422A;
	mul.f32 	%r1055, %r1053, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1056, %r1046, 0f3F000000;
	mul.f32 	%r1057, %r1047, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1058, %r1054, 0fC0000000;
	mul.f32 	%r1059, %r1055, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1060, %r1058, 0f3FB8AA3B;
	ex2.approx.f32 	%r1061, %r1060;
	mul.f32 	%r1062, %r1059, 0f3FB8AA3B;
	ex2.approx.f32 	%r1063, %r1062;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1064, %r1061, 0f3F800000;
	add.f32 	%r1065, %r1063, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1066, %r491, %r1064;
	div.full.f32 	%r1067, %r491, %r1065;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1068, %r1066, 0fBF800000;
	add.f32 	%r1069, %r1067, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1070, %r1069, 0f3F800000;
	add.f32 	%r1071, %r1068, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1072, %r1057, %r1071;
	mul.f32 	%r1073, %r1056, %r1070;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r306, %r1073, %r1072;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1074, %r2347, %r617;
	add.f32 	%r1075, %r2343, %r616;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1076, %r1075, 0f3D372713;
	mul.f32 	%r1077, %r1074, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1078, %r1075, %r1076;
	mul.f32 	%r1079, %r1074, %r1077;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1080, %r1075, %r1078, %r1075;
	fma.rn.f32 	%r1081, %r1074, %r1079, %r1074;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1082, %r1080, 0f3F4C422A;
	mul.f32 	%r1083, %r1081, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1084, %r1074, 0f3F000000;
	mul.f32 	%r1085, %r1075, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1086, %r1082, 0fC0000000;
	mul.f32 	%r1087, %r1083, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1088, %r1086, 0f3FB8AA3B;
	ex2.approx.f32 	%r1089, %r1088;
	mul.f32 	%r1090, %r1087, 0f3FB8AA3B;
	ex2.approx.f32 	%r1091, %r1090;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1092, %r1089, 0f3F800000;
	add.f32 	%r1093, %r1091, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1094, %r491, %r1092;
	div.full.f32 	%r1095, %r491, %r1093;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1096, %r1094, 0fBF800000;
	add.f32 	%r1097, %r1095, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1098, %r1097, 0f3F800000;
	add.f32 	%r1099, %r1096, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1100, %r1085, %r1099;
	mul.f32 	%r1101, %r1084, %r1098;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r310, %r1101, %r1100;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1102, %r2348, %r587;
	add.f32 	%r1103, %r2344, %r586;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1104, %r1103, 0f3D372713;
	mul.f32 	%r1105, %r1102, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1106, %r1103, %r1104;
	mul.f32 	%r1107, %r1102, %r1105;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1108, %r1103, %r1106, %r1103;
	fma.rn.f32 	%r1109, %r1102, %r1107, %r1102;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1110, %r1108, 0f3F4C422A;
	mul.f32 	%r1111, %r1109, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1112, %r1102, 0f3F000000;
	mul.f32 	%r1113, %r1103, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1114, %r1110, 0fC0000000;
	mul.f32 	%r1115, %r1111, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1116, %r1114, 0f3FB8AA3B;
	ex2.approx.f32 	%r1117, %r1116;
	mul.f32 	%r1118, %r1115, 0f3FB8AA3B;
	ex2.approx.f32 	%r1119, %r1118;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1120, %r1117, 0f3F800000;
	add.f32 	%r1121, %r1119, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1122, %r491, %r1120;
	div.full.f32 	%r1123, %r491, %r1121;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1124, %r1122, 0fBF800000;
	add.f32 	%r1125, %r1123, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1126, %r1125, 0f3F800000;
	add.f32 	%r1127, %r1124, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1128, %r1113, %r1127;
	mul.f32 	%r1129, %r1112, %r1126;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r314, %r1129, %r1128;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1130, %r2349, %r617;
	add.f32 	%r1131, %r2345, %r616;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1132, %r1131, 0f3D372713;
	mul.f32 	%r1133, %r1130, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1134, %r1131, %r1132;
	mul.f32 	%r1135, %r1130, %r1133;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1136, %r1131, %r1134, %r1131;
	fma.rn.f32 	%r1137, %r1130, %r1135, %r1130;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1138, %r1136, 0f3F4C422A;
	mul.f32 	%r1139, %r1137, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1140, %r1130, 0f3F000000;
	mul.f32 	%r1141, %r1131, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1142, %r1138, 0fC0000000;
	mul.f32 	%r1143, %r1139, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1144, %r1142, 0f3FB8AA3B;
	ex2.approx.f32 	%r1145, %r1144;
	mul.f32 	%r1146, %r1143, 0f3FB8AA3B;
	ex2.approx.f32 	%r1147, %r1146;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1148, %r1145, 0f3F800000;
	add.f32 	%r1149, %r1147, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1150, %r491, %r1148;
	div.full.f32 	%r1151, %r491, %r1149;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1152, %r1150, 0fBF800000;
	add.f32 	%r1153, %r1151, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1154, %r1153, 0f3F800000;
	add.f32 	%r1155, %r1152, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1156, %r1141, %r1155;
	mul.f32 	%r1157, %r1140, %r1154;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r318, %r1157, %r1156;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1158, %r2354, %r703;
	add.f32 	%r1159, %r2350, %r702;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1160, %r1159, 0f3D372713;
	mul.f32 	%r1161, %r1158, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1162, %r1159, %r1160;
	mul.f32 	%r1163, %r1158, %r1161;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1164, %r1159, %r1162, %r1159;
	fma.rn.f32 	%r1165, %r1158, %r1163, %r1158;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1166, %r1164, 0f3F4C422A;
	mul.f32 	%r1167, %r1165, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1168, %r1158, 0f3F000000;
	mul.f32 	%r1169, %r1159, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1170, %r1166, 0fC0000000;
	mul.f32 	%r1171, %r1167, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1172, %r1170, 0f3FB8AA3B;
	ex2.approx.f32 	%r1173, %r1172;
	mul.f32 	%r1174, %r1171, 0f3FB8AA3B;
	ex2.approx.f32 	%r1175, %r1174;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1176, %r1173, 0f3F800000;
	add.f32 	%r1177, %r1175, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1178, %r491, %r1176;
	div.full.f32 	%r1179, %r491, %r1177;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1180, %r1178, 0fBF800000;
	add.f32 	%r1181, %r1179, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1182, %r1181, 0f3F800000;
	add.f32 	%r1183, %r1180, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1184, %r1169, %r1183;
	mul.f32 	%r1185, %r1168, %r1182;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r307, %r1185, %r1184;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1186, %r2355, %r733;
	add.f32 	%r1187, %r2351, %r732;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1188, %r1187, 0f3D372713;
	mul.f32 	%r1189, %r1186, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1190, %r1187, %r1188;
	mul.f32 	%r1191, %r1186, %r1189;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1192, %r1187, %r1190, %r1187;
	fma.rn.f32 	%r1193, %r1186, %r1191, %r1186;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1194, %r1192, 0f3F4C422A;
	mul.f32 	%r1195, %r1193, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1196, %r1186, 0f3F000000;
	mul.f32 	%r1197, %r1187, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1198, %r1194, 0fC0000000;
	mul.f32 	%r1199, %r1195, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1200, %r1198, 0f3FB8AA3B;
	ex2.approx.f32 	%r1201, %r1200;
	mul.f32 	%r1202, %r1199, 0f3FB8AA3B;
	ex2.approx.f32 	%r1203, %r1202;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1204, %r1201, 0f3F800000;
	add.f32 	%r1205, %r1203, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1206, %r491, %r1204;
	div.full.f32 	%r1207, %r491, %r1205;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1208, %r1206, 0fBF800000;
	add.f32 	%r1209, %r1207, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1210, %r1209, 0f3F800000;
	add.f32 	%r1211, %r1208, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1212, %r1197, %r1211;
	mul.f32 	%r1213, %r1196, %r1210;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r311, %r1213, %r1212;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1214, %r2356, %r703;
	add.f32 	%r1215, %r2352, %r702;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1216, %r1215, 0f3D372713;
	mul.f32 	%r1217, %r1214, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1218, %r1215, %r1216;
	mul.f32 	%r1219, %r1214, %r1217;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1220, %r1215, %r1218, %r1215;
	fma.rn.f32 	%r1221, %r1214, %r1219, %r1214;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1222, %r1220, 0f3F4C422A;
	mul.f32 	%r1223, %r1221, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1224, %r1214, 0f3F000000;
	mul.f32 	%r1225, %r1215, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1226, %r1222, 0fC0000000;
	mul.f32 	%r1227, %r1223, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1228, %r1226, 0f3FB8AA3B;
	ex2.approx.f32 	%r1229, %r1228;
	mul.f32 	%r1230, %r1227, 0f3FB8AA3B;
	ex2.approx.f32 	%r1231, %r1230;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1232, %r1229, 0f3F800000;
	add.f32 	%r1233, %r1231, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1234, %r491, %r1232;
	div.full.f32 	%r1235, %r491, %r1233;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1236, %r1234, 0fBF800000;
	add.f32 	%r1237, %r1235, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1238, %r1237, 0f3F800000;
	add.f32 	%r1239, %r1236, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1240, %r1225, %r1239;
	mul.f32 	%r1241, %r1224, %r1238;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r315, %r1241, %r1240;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1242, %r2357, %r733;
	add.f32 	%r1243, %r2353, %r732;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1244, %r1243, 0f3D372713;
	mul.f32 	%r1245, %r1242, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1246, %r1243, %r1244;
	mul.f32 	%r1247, %r1242, %r1245;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1248, %r1243, %r1246, %r1243;
	fma.rn.f32 	%r1249, %r1242, %r1247, %r1242;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1250, %r1248, 0f3F4C422A;
	mul.f32 	%r1251, %r1249, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1252, %r1242, 0f3F000000;
	mul.f32 	%r1253, %r1243, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1254, %r1250, 0fC0000000;
	mul.f32 	%r1255, %r1251, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1256, %r1254, 0f3FB8AA3B;
	ex2.approx.f32 	%r1257, %r1256;
	mul.f32 	%r1258, %r1255, 0f3FB8AA3B;
	ex2.approx.f32 	%r1259, %r1258;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1260, %r1257, 0f3F800000;
	add.f32 	%r1261, %r1259, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1262, %r491, %r1260;
	div.full.f32 	%r1263, %r491, %r1261;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1264, %r1262, 0fBF800000;
	add.f32 	%r1265, %r1263, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1266, %r1265, 0f3F800000;
	add.f32 	%r1267, %r1264, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1268, %r1253, %r1267;
	mul.f32 	%r1269, %r1252, %r1266;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r319, %r1269, %r1268;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1270, %r2362, %r819;
	add.f32 	%r1271, %r2358, %r818;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1272, %r1271, 0f3D372713;
	mul.f32 	%r1273, %r1270, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1274, %r1271, %r1272;
	mul.f32 	%r1275, %r1270, %r1273;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1276, %r1271, %r1274, %r1271;
	fma.rn.f32 	%r1277, %r1270, %r1275, %r1270;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1278, %r1276, 0f3F4C422A;
	mul.f32 	%r1279, %r1277, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1280, %r1270, 0f3F000000;
	mul.f32 	%r1281, %r1271, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1282, %r1278, 0fC0000000;
	mul.f32 	%r1283, %r1279, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1284, %r1282, 0f3FB8AA3B;
	ex2.approx.f32 	%r1285, %r1284;
	mul.f32 	%r1286, %r1283, 0f3FB8AA3B;
	ex2.approx.f32 	%r1287, %r1286;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1288, %r1285, 0f3F800000;
	add.f32 	%r1289, %r1287, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1290, %r491, %r1288;
	div.full.f32 	%r1291, %r491, %r1289;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1292, %r1290, 0fBF800000;
	add.f32 	%r1293, %r1291, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1294, %r1293, 0f3F800000;
	add.f32 	%r1295, %r1292, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1296, %r1281, %r1295;
	mul.f32 	%r1297, %r1280, %r1294;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r308, %r1297, %r1296;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1298, %r2363, %r849;
	add.f32 	%r1299, %r2359, %r848;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1300, %r1299, 0f3D372713;
	mul.f32 	%r1301, %r1298, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1302, %r1299, %r1300;
	mul.f32 	%r1303, %r1298, %r1301;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1304, %r1299, %r1302, %r1299;
	fma.rn.f32 	%r1305, %r1298, %r1303, %r1298;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1306, %r1304, 0f3F4C422A;
	mul.f32 	%r1307, %r1305, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1308, %r1298, 0f3F000000;
	mul.f32 	%r1309, %r1299, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1310, %r1306, 0fC0000000;
	mul.f32 	%r1311, %r1307, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1312, %r1310, 0f3FB8AA3B;
	ex2.approx.f32 	%r1313, %r1312;
	mul.f32 	%r1314, %r1311, 0f3FB8AA3B;
	ex2.approx.f32 	%r1315, %r1314;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1316, %r1313, 0f3F800000;
	add.f32 	%r1317, %r1315, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1318, %r491, %r1316;
	div.full.f32 	%r1319, %r491, %r1317;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1320, %r1318, 0fBF800000;
	add.f32 	%r1321, %r1319, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1322, %r1321, 0f3F800000;
	add.f32 	%r1323, %r1320, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1324, %r1309, %r1323;
	mul.f32 	%r1325, %r1308, %r1322;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r312, %r1325, %r1324;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1326, %r2364, %r819;
	add.f32 	%r1327, %r2360, %r818;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1328, %r1327, 0f3D372713;
	mul.f32 	%r1329, %r1326, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1330, %r1327, %r1328;
	mul.f32 	%r1331, %r1326, %r1329;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1332, %r1327, %r1330, %r1327;
	fma.rn.f32 	%r1333, %r1326, %r1331, %r1326;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1334, %r1332, 0f3F4C422A;
	mul.f32 	%r1335, %r1333, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1336, %r1326, 0f3F000000;
	mul.f32 	%r1337, %r1327, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1338, %r1334, 0fC0000000;
	mul.f32 	%r1339, %r1335, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1340, %r1338, 0f3FB8AA3B;
	ex2.approx.f32 	%r1341, %r1340;
	mul.f32 	%r1342, %r1339, 0f3FB8AA3B;
	ex2.approx.f32 	%r1343, %r1342;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1344, %r1341, 0f3F800000;
	add.f32 	%r1345, %r1343, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1346, %r491, %r1344;
	div.full.f32 	%r1347, %r491, %r1345;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1348, %r1346, 0fBF800000;
	add.f32 	%r1349, %r1347, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1350, %r1349, 0f3F800000;
	add.f32 	%r1351, %r1348, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1352, %r1337, %r1351;
	mul.f32 	%r1353, %r1336, %r1350;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r316, %r1353, %r1352;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1354, %r2365, %r849;
	add.f32 	%r1355, %r2361, %r848;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1356, %r1355, 0f3D372713;
	mul.f32 	%r1357, %r1354, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1358, %r1355, %r1356;
	mul.f32 	%r1359, %r1354, %r1357;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1360, %r1355, %r1358, %r1355;
	fma.rn.f32 	%r1361, %r1354, %r1359, %r1354;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1362, %r1360, 0f3F4C422A;
	mul.f32 	%r1363, %r1361, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1364, %r1354, 0f3F000000;
	mul.f32 	%r1365, %r1355, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1366, %r1362, 0fC0000000;
	mul.f32 	%r1367, %r1363, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1368, %r1366, 0f3FB8AA3B;
	ex2.approx.f32 	%r1369, %r1368;
	mul.f32 	%r1370, %r1367, 0f3FB8AA3B;
	ex2.approx.f32 	%r1371, %r1370;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1372, %r1369, 0f3F800000;
	add.f32 	%r1373, %r1371, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1374, %r491, %r1372;
	div.full.f32 	%r1375, %r491, %r1373;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1376, %r1374, 0fBF800000;
	add.f32 	%r1377, %r1375, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1378, %r1377, 0f3F800000;
	add.f32 	%r1379, %r1376, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1380, %r1365, %r1379;
	mul.f32 	%r1381, %r1364, %r1378;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r320, %r1381, %r1380;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1382, %r2370, %r470;
	add.f32 	%r1383, %r2366, %r469;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1384, %r1383, 0f3D372713;
	mul.f32 	%r1385, %r1382, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1386, %r1383, %r1384;
	mul.f32 	%r1387, %r1382, %r1385;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1388, %r1383, %r1386, %r1383;
	fma.rn.f32 	%r1389, %r1382, %r1387, %r1382;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1390, %r1388, 0f3F4C422A;
	mul.f32 	%r1391, %r1389, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1392, %r1382, 0f3F000000;
	mul.f32 	%r1393, %r1383, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1394, %r1390, 0fC0000000;
	mul.f32 	%r1395, %r1391, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1396, %r1394, 0f3FB8AA3B;
	ex2.approx.f32 	%r1397, %r1396;
	mul.f32 	%r1398, %r1395, 0f3FB8AA3B;
	ex2.approx.f32 	%r1399, %r1398;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1400, %r1397, 0f3F800000;
	add.f32 	%r1401, %r1399, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1402, %r491, %r1400;
	div.full.f32 	%r1403, %r491, %r1401;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1404, %r1402, 0fBF800000;
	add.f32 	%r1405, %r1403, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1406, %r1405, 0f3F800000;
	add.f32 	%r1407, %r1404, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1408, %r1393, %r1407;
	mul.f32 	%r1409, %r1392, %r1406;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r321, %r1409, %r1408;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1410, %r2371, %r501;
	add.f32 	%r1411, %r2367, %r500;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1412, %r1411, 0f3D372713;
	mul.f32 	%r1413, %r1410, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1414, %r1411, %r1412;
	mul.f32 	%r1415, %r1410, %r1413;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1416, %r1411, %r1414, %r1411;
	fma.rn.f32 	%r1417, %r1410, %r1415, %r1410;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1418, %r1416, 0f3F4C422A;
	mul.f32 	%r1419, %r1417, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1420, %r1410, 0f3F000000;
	mul.f32 	%r1421, %r1411, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1422, %r1418, 0fC0000000;
	mul.f32 	%r1423, %r1419, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1424, %r1422, 0f3FB8AA3B;
	ex2.approx.f32 	%r1425, %r1424;
	mul.f32 	%r1426, %r1423, 0f3FB8AA3B;
	ex2.approx.f32 	%r1427, %r1426;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1428, %r1425, 0f3F800000;
	add.f32 	%r1429, %r1427, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1430, %r491, %r1428;
	div.full.f32 	%r1431, %r491, %r1429;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1432, %r1430, 0fBF800000;
	add.f32 	%r1433, %r1431, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1434, %r1433, 0f3F800000;
	add.f32 	%r1435, %r1432, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1436, %r1421, %r1435;
	mul.f32 	%r1437, %r1420, %r1434;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r325, %r1437, %r1436;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1438, %r2372, %r470;
	add.f32 	%r1439, %r2368, %r469;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1440, %r1439, 0f3D372713;
	mul.f32 	%r1441, %r1438, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1442, %r1439, %r1440;
	mul.f32 	%r1443, %r1438, %r1441;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1444, %r1439, %r1442, %r1439;
	fma.rn.f32 	%r1445, %r1438, %r1443, %r1438;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1446, %r1444, 0f3F4C422A;
	mul.f32 	%r1447, %r1445, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1448, %r1438, 0f3F000000;
	mul.f32 	%r1449, %r1439, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1450, %r1446, 0fC0000000;
	mul.f32 	%r1451, %r1447, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1452, %r1450, 0f3FB8AA3B;
	ex2.approx.f32 	%r1453, %r1452;
	mul.f32 	%r1454, %r1451, 0f3FB8AA3B;
	ex2.approx.f32 	%r1455, %r1454;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1456, %r1453, 0f3F800000;
	add.f32 	%r1457, %r1455, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1458, %r491, %r1456;
	div.full.f32 	%r1459, %r491, %r1457;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1460, %r1458, 0fBF800000;
	add.f32 	%r1461, %r1459, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1462, %r1461, 0f3F800000;
	add.f32 	%r1463, %r1460, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1464, %r1449, %r1463;
	mul.f32 	%r1465, %r1448, %r1462;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r329, %r1465, %r1464;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1466, %r2373, %r501;
	add.f32 	%r1467, %r2369, %r500;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1468, %r1467, 0f3D372713;
	mul.f32 	%r1469, %r1466, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1470, %r1467, %r1468;
	mul.f32 	%r1471, %r1466, %r1469;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1472, %r1467, %r1470, %r1467;
	fma.rn.f32 	%r1473, %r1466, %r1471, %r1466;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1474, %r1472, 0f3F4C422A;
	mul.f32 	%r1475, %r1473, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1476, %r1466, 0f3F000000;
	mul.f32 	%r1477, %r1467, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1478, %r1474, 0fC0000000;
	mul.f32 	%r1479, %r1475, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1480, %r1478, 0f3FB8AA3B;
	ex2.approx.f32 	%r1481, %r1480;
	mul.f32 	%r1482, %r1479, 0f3FB8AA3B;
	ex2.approx.f32 	%r1483, %r1482;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1484, %r1481, 0f3F800000;
	add.f32 	%r1485, %r1483, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1486, %r491, %r1484;
	div.full.f32 	%r1487, %r491, %r1485;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1488, %r1486, 0fBF800000;
	add.f32 	%r1489, %r1487, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1490, %r1489, 0f3F800000;
	add.f32 	%r1491, %r1488, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1492, %r1477, %r1491;
	mul.f32 	%r1493, %r1476, %r1490;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r333, %r1493, %r1492;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1494, %r2378, %r587;
	add.f32 	%r1495, %r2374, %r586;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1496, %r1495, 0f3D372713;
	mul.f32 	%r1497, %r1494, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1498, %r1495, %r1496;
	mul.f32 	%r1499, %r1494, %r1497;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1500, %r1495, %r1498, %r1495;
	fma.rn.f32 	%r1501, %r1494, %r1499, %r1494;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1502, %r1500, 0f3F4C422A;
	mul.f32 	%r1503, %r1501, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1504, %r1494, 0f3F000000;
	mul.f32 	%r1505, %r1495, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1506, %r1502, 0fC0000000;
	mul.f32 	%r1507, %r1503, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1508, %r1506, 0f3FB8AA3B;
	ex2.approx.f32 	%r1509, %r1508;
	mul.f32 	%r1510, %r1507, 0f3FB8AA3B;
	ex2.approx.f32 	%r1511, %r1510;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1512, %r1509, 0f3F800000;
	add.f32 	%r1513, %r1511, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1514, %r491, %r1512;
	div.full.f32 	%r1515, %r491, %r1513;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1516, %r1514, 0fBF800000;
	add.f32 	%r1517, %r1515, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1518, %r1517, 0f3F800000;
	add.f32 	%r1519, %r1516, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1520, %r1505, %r1519;
	mul.f32 	%r1521, %r1504, %r1518;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r322, %r1521, %r1520;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1522, %r2379, %r617;
	add.f32 	%r1523, %r2375, %r616;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1524, %r1523, 0f3D372713;
	mul.f32 	%r1525, %r1522, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1526, %r1523, %r1524;
	mul.f32 	%r1527, %r1522, %r1525;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1528, %r1523, %r1526, %r1523;
	fma.rn.f32 	%r1529, %r1522, %r1527, %r1522;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1530, %r1528, 0f3F4C422A;
	mul.f32 	%r1531, %r1529, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1532, %r1522, 0f3F000000;
	mul.f32 	%r1533, %r1523, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1534, %r1530, 0fC0000000;
	mul.f32 	%r1535, %r1531, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1536, %r1534, 0f3FB8AA3B;
	ex2.approx.f32 	%r1537, %r1536;
	mul.f32 	%r1538, %r1535, 0f3FB8AA3B;
	ex2.approx.f32 	%r1539, %r1538;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1540, %r1537, 0f3F800000;
	add.f32 	%r1541, %r1539, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1542, %r491, %r1540;
	div.full.f32 	%r1543, %r491, %r1541;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1544, %r1542, 0fBF800000;
	add.f32 	%r1545, %r1543, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1546, %r1545, 0f3F800000;
	add.f32 	%r1547, %r1544, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1548, %r1533, %r1547;
	mul.f32 	%r1549, %r1532, %r1546;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r326, %r1549, %r1548;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1550, %r2380, %r587;
	add.f32 	%r1551, %r2376, %r586;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1552, %r1551, 0f3D372713;
	mul.f32 	%r1553, %r1550, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1554, %r1551, %r1552;
	mul.f32 	%r1555, %r1550, %r1553;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1556, %r1551, %r1554, %r1551;
	fma.rn.f32 	%r1557, %r1550, %r1555, %r1550;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1558, %r1556, 0f3F4C422A;
	mul.f32 	%r1559, %r1557, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1560, %r1550, 0f3F000000;
	mul.f32 	%r1561, %r1551, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1562, %r1558, 0fC0000000;
	mul.f32 	%r1563, %r1559, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1564, %r1562, 0f3FB8AA3B;
	ex2.approx.f32 	%r1565, %r1564;
	mul.f32 	%r1566, %r1563, 0f3FB8AA3B;
	ex2.approx.f32 	%r1567, %r1566;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1568, %r1565, 0f3F800000;
	add.f32 	%r1569, %r1567, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1570, %r491, %r1568;
	div.full.f32 	%r1571, %r491, %r1569;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1572, %r1570, 0fBF800000;
	add.f32 	%r1573, %r1571, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1574, %r1573, 0f3F800000;
	add.f32 	%r1575, %r1572, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1576, %r1561, %r1575;
	mul.f32 	%r1577, %r1560, %r1574;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r330, %r1577, %r1576;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1578, %r2381, %r617;
	add.f32 	%r1579, %r2377, %r616;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1580, %r1579, 0f3D372713;
	mul.f32 	%r1581, %r1578, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1582, %r1579, %r1580;
	mul.f32 	%r1583, %r1578, %r1581;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1584, %r1579, %r1582, %r1579;
	fma.rn.f32 	%r1585, %r1578, %r1583, %r1578;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1586, %r1584, 0f3F4C422A;
	mul.f32 	%r1587, %r1585, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1588, %r1578, 0f3F000000;
	mul.f32 	%r1589, %r1579, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1590, %r1586, 0fC0000000;
	mul.f32 	%r1591, %r1587, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1592, %r1590, 0f3FB8AA3B;
	ex2.approx.f32 	%r1593, %r1592;
	mul.f32 	%r1594, %r1591, 0f3FB8AA3B;
	ex2.approx.f32 	%r1595, %r1594;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1596, %r1593, 0f3F800000;
	add.f32 	%r1597, %r1595, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1598, %r491, %r1596;
	div.full.f32 	%r1599, %r491, %r1597;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1600, %r1598, 0fBF800000;
	add.f32 	%r1601, %r1599, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1602, %r1601, 0f3F800000;
	add.f32 	%r1603, %r1600, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1604, %r1589, %r1603;
	mul.f32 	%r1605, %r1588, %r1602;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r334, %r1605, %r1604;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1606, %r2386, %r703;
	add.f32 	%r1607, %r2382, %r702;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1608, %r1607, 0f3D372713;
	mul.f32 	%r1609, %r1606, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1610, %r1607, %r1608;
	mul.f32 	%r1611, %r1606, %r1609;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1612, %r1607, %r1610, %r1607;
	fma.rn.f32 	%r1613, %r1606, %r1611, %r1606;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1614, %r1612, 0f3F4C422A;
	mul.f32 	%r1615, %r1613, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1616, %r1606, 0f3F000000;
	mul.f32 	%r1617, %r1607, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1618, %r1614, 0fC0000000;
	mul.f32 	%r1619, %r1615, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1620, %r1618, 0f3FB8AA3B;
	ex2.approx.f32 	%r1621, %r1620;
	mul.f32 	%r1622, %r1619, 0f3FB8AA3B;
	ex2.approx.f32 	%r1623, %r1622;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1624, %r1621, 0f3F800000;
	add.f32 	%r1625, %r1623, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1626, %r491, %r1624;
	div.full.f32 	%r1627, %r491, %r1625;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1628, %r1626, 0fBF800000;
	add.f32 	%r1629, %r1627, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1630, %r1629, 0f3F800000;
	add.f32 	%r1631, %r1628, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1632, %r1617, %r1631;
	mul.f32 	%r1633, %r1616, %r1630;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r323, %r1633, %r1632;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1634, %r2387, %r733;
	add.f32 	%r1635, %r2383, %r732;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1636, %r1635, 0f3D372713;
	mul.f32 	%r1637, %r1634, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1638, %r1635, %r1636;
	mul.f32 	%r1639, %r1634, %r1637;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1640, %r1635, %r1638, %r1635;
	fma.rn.f32 	%r1641, %r1634, %r1639, %r1634;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1642, %r1640, 0f3F4C422A;
	mul.f32 	%r1643, %r1641, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1644, %r1634, 0f3F000000;
	mul.f32 	%r1645, %r1635, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1646, %r1642, 0fC0000000;
	mul.f32 	%r1647, %r1643, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1648, %r1646, 0f3FB8AA3B;
	ex2.approx.f32 	%r1649, %r1648;
	mul.f32 	%r1650, %r1647, 0f3FB8AA3B;
	ex2.approx.f32 	%r1651, %r1650;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1652, %r1649, 0f3F800000;
	add.f32 	%r1653, %r1651, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1654, %r491, %r1652;
	div.full.f32 	%r1655, %r491, %r1653;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1656, %r1654, 0fBF800000;
	add.f32 	%r1657, %r1655, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1658, %r1657, 0f3F800000;
	add.f32 	%r1659, %r1656, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1660, %r1645, %r1659;
	mul.f32 	%r1661, %r1644, %r1658;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r327, %r1661, %r1660;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1662, %r2388, %r703;
	add.f32 	%r1663, %r2384, %r702;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1664, %r1663, 0f3D372713;
	mul.f32 	%r1665, %r1662, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1666, %r1663, %r1664;
	mul.f32 	%r1667, %r1662, %r1665;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1668, %r1663, %r1666, %r1663;
	fma.rn.f32 	%r1669, %r1662, %r1667, %r1662;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1670, %r1668, 0f3F4C422A;
	mul.f32 	%r1671, %r1669, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1672, %r1662, 0f3F000000;
	mul.f32 	%r1673, %r1663, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1674, %r1670, 0fC0000000;
	mul.f32 	%r1675, %r1671, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1676, %r1674, 0f3FB8AA3B;
	ex2.approx.f32 	%r1677, %r1676;
	mul.f32 	%r1678, %r1675, 0f3FB8AA3B;
	ex2.approx.f32 	%r1679, %r1678;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1680, %r1677, 0f3F800000;
	add.f32 	%r1681, %r1679, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1682, %r491, %r1680;
	div.full.f32 	%r1683, %r491, %r1681;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1684, %r1682, 0fBF800000;
	add.f32 	%r1685, %r1683, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1686, %r1685, 0f3F800000;
	add.f32 	%r1687, %r1684, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1688, %r1673, %r1687;
	mul.f32 	%r1689, %r1672, %r1686;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r331, %r1689, %r1688;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1690, %r2389, %r733;
	add.f32 	%r1691, %r2385, %r732;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1692, %r1691, 0f3D372713;
	mul.f32 	%r1693, %r1690, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1694, %r1691, %r1692;
	mul.f32 	%r1695, %r1690, %r1693;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1696, %r1691, %r1694, %r1691;
	fma.rn.f32 	%r1697, %r1690, %r1695, %r1690;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1698, %r1696, 0f3F4C422A;
	mul.f32 	%r1699, %r1697, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1700, %r1690, 0f3F000000;
	mul.f32 	%r1701, %r1691, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1702, %r1698, 0fC0000000;
	mul.f32 	%r1703, %r1699, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1704, %r1702, 0f3FB8AA3B;
	ex2.approx.f32 	%r1705, %r1704;
	mul.f32 	%r1706, %r1703, 0f3FB8AA3B;
	ex2.approx.f32 	%r1707, %r1706;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1708, %r1705, 0f3F800000;
	add.f32 	%r1709, %r1707, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1710, %r491, %r1708;
	div.full.f32 	%r1711, %r491, %r1709;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1712, %r1710, 0fBF800000;
	add.f32 	%r1713, %r1711, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1714, %r1713, 0f3F800000;
	add.f32 	%r1715, %r1712, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1716, %r1701, %r1715;
	mul.f32 	%r1717, %r1700, %r1714;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r335, %r1717, %r1716;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1718, %r2394, %r819;
	add.f32 	%r1719, %r2390, %r818;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1720, %r1719, 0f3D372713;
	mul.f32 	%r1721, %r1718, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1722, %r1719, %r1720;
	mul.f32 	%r1723, %r1718, %r1721;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1724, %r1719, %r1722, %r1719;
	fma.rn.f32 	%r1725, %r1718, %r1723, %r1718;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1726, %r1724, 0f3F4C422A;
	mul.f32 	%r1727, %r1725, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1728, %r1718, 0f3F000000;
	mul.f32 	%r1729, %r1719, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1730, %r1726, 0fC0000000;
	mul.f32 	%r1731, %r1727, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1732, %r1730, 0f3FB8AA3B;
	ex2.approx.f32 	%r1733, %r1732;
	mul.f32 	%r1734, %r1731, 0f3FB8AA3B;
	ex2.approx.f32 	%r1735, %r1734;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1736, %r1733, 0f3F800000;
	add.f32 	%r1737, %r1735, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1738, %r491, %r1736;
	div.full.f32 	%r1739, %r491, %r1737;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1740, %r1738, 0fBF800000;
	add.f32 	%r1741, %r1739, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1742, %r1741, 0f3F800000;
	add.f32 	%r1743, %r1740, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1744, %r1729, %r1743;
	mul.f32 	%r1745, %r1728, %r1742;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r324, %r1745, %r1744;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1746, %r2395, %r849;
	add.f32 	%r1747, %r2391, %r848;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1748, %r1747, 0f3D372713;
	mul.f32 	%r1749, %r1746, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1750, %r1747, %r1748;
	mul.f32 	%r1751, %r1746, %r1749;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1752, %r1747, %r1750, %r1747;
	fma.rn.f32 	%r1753, %r1746, %r1751, %r1746;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1754, %r1752, 0f3F4C422A;
	mul.f32 	%r1755, %r1753, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1756, %r1746, 0f3F000000;
	mul.f32 	%r1757, %r1747, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1758, %r1754, 0fC0000000;
	mul.f32 	%r1759, %r1755, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1760, %r1758, 0f3FB8AA3B;
	ex2.approx.f32 	%r1761, %r1760;
	mul.f32 	%r1762, %r1759, 0f3FB8AA3B;
	ex2.approx.f32 	%r1763, %r1762;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1764, %r1761, 0f3F800000;
	add.f32 	%r1765, %r1763, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1766, %r491, %r1764;
	div.full.f32 	%r1767, %r491, %r1765;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1768, %r1766, 0fBF800000;
	add.f32 	%r1769, %r1767, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1770, %r1769, 0f3F800000;
	add.f32 	%r1771, %r1768, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1772, %r1757, %r1771;
	mul.f32 	%r1773, %r1756, %r1770;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r328, %r1773, %r1772;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1774, %r2396, %r819;
	add.f32 	%r1775, %r2392, %r818;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1776, %r1775, 0f3D372713;
	mul.f32 	%r1777, %r1774, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1778, %r1775, %r1776;
	mul.f32 	%r1779, %r1774, %r1777;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1780, %r1775, %r1778, %r1775;
	fma.rn.f32 	%r1781, %r1774, %r1779, %r1774;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1782, %r1780, 0f3F4C422A;
	mul.f32 	%r1783, %r1781, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1784, %r1774, 0f3F000000;
	mul.f32 	%r1785, %r1775, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1786, %r1782, 0fC0000000;
	mul.f32 	%r1787, %r1783, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1788, %r1786, 0f3FB8AA3B;
	ex2.approx.f32 	%r1789, %r1788;
	mul.f32 	%r1790, %r1787, 0f3FB8AA3B;
	ex2.approx.f32 	%r1791, %r1790;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1792, %r1789, 0f3F800000;
	add.f32 	%r1793, %r1791, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1794, %r491, %r1792;
	div.full.f32 	%r1795, %r491, %r1793;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1796, %r1794, 0fBF800000;
	add.f32 	%r1797, %r1795, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1798, %r1797, 0f3F800000;
	add.f32 	%r1799, %r1796, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1800, %r1785, %r1799;
	mul.f32 	%r1801, %r1784, %r1798;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r332, %r1801, %r1800;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1802, %r2397, %r849;
	add.f32 	%r1803, %r2393, %r848;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1804, %r1803, 0f3D372713;
	mul.f32 	%r1805, %r1802, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1806, %r1803, %r1804;
	mul.f32 	%r1807, %r1802, %r1805;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1808, %r1803, %r1806, %r1803;
	fma.rn.f32 	%r1809, %r1802, %r1807, %r1802;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1810, %r1808, 0f3F4C422A;
	mul.f32 	%r1811, %r1809, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1812, %r1802, 0f3F000000;
	mul.f32 	%r1813, %r1803, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1814, %r1810, 0fC0000000;
	mul.f32 	%r1815, %r1811, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1816, %r1814, 0f3FB8AA3B;
	ex2.approx.f32 	%r1817, %r1816;
	mul.f32 	%r1818, %r1815, 0f3FB8AA3B;
	ex2.approx.f32 	%r1819, %r1818;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1820, %r1817, 0f3F800000;
	add.f32 	%r1821, %r1819, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1822, %r491, %r1820;
	div.full.f32 	%r1823, %r491, %r1821;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1824, %r1822, 0fBF800000;
	add.f32 	%r1825, %r1823, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1826, %r1825, 0f3F800000;
	add.f32 	%r1827, %r1824, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1828, %r1813, %r1827;
	mul.f32 	%r1829, %r1812, %r1826;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r336, %r1829, %r1828;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1830, %r2402, %r470;
	add.f32 	%r1831, %r2398, %r469;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1832, %r1831, 0f3D372713;
	mul.f32 	%r1833, %r1830, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1834, %r1831, %r1832;
	mul.f32 	%r1835, %r1830, %r1833;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1836, %r1831, %r1834, %r1831;
	fma.rn.f32 	%r1837, %r1830, %r1835, %r1830;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1838, %r1836, 0f3F4C422A;
	mul.f32 	%r1839, %r1837, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1840, %r1830, 0f3F000000;
	mul.f32 	%r1841, %r1831, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1842, %r1838, 0fC0000000;
	mul.f32 	%r1843, %r1839, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1844, %r1842, 0f3FB8AA3B;
	ex2.approx.f32 	%r1845, %r1844;
	mul.f32 	%r1846, %r1843, 0f3FB8AA3B;
	ex2.approx.f32 	%r1847, %r1846;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1848, %r1845, 0f3F800000;
	add.f32 	%r1849, %r1847, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1850, %r491, %r1848;
	div.full.f32 	%r1851, %r491, %r1849;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1852, %r1850, 0fBF800000;
	add.f32 	%r1853, %r1851, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1854, %r1853, 0f3F800000;
	add.f32 	%r1855, %r1852, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1856, %r1841, %r1855;
	mul.f32 	%r1857, %r1840, %r1854;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r337, %r1857, %r1856;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1858, %r2403, %r501;
	add.f32 	%r1859, %r2399, %r500;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1860, %r1859, 0f3D372713;
	mul.f32 	%r1861, %r1858, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1862, %r1859, %r1860;
	mul.f32 	%r1863, %r1858, %r1861;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1864, %r1859, %r1862, %r1859;
	fma.rn.f32 	%r1865, %r1858, %r1863, %r1858;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1866, %r1864, 0f3F4C422A;
	mul.f32 	%r1867, %r1865, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1868, %r1858, 0f3F000000;
	mul.f32 	%r1869, %r1859, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1870, %r1866, 0fC0000000;
	mul.f32 	%r1871, %r1867, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1872, %r1870, 0f3FB8AA3B;
	ex2.approx.f32 	%r1873, %r1872;
	mul.f32 	%r1874, %r1871, 0f3FB8AA3B;
	ex2.approx.f32 	%r1875, %r1874;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1876, %r1873, 0f3F800000;
	add.f32 	%r1877, %r1875, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1878, %r491, %r1876;
	div.full.f32 	%r1879, %r491, %r1877;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1880, %r1878, 0fBF800000;
	add.f32 	%r1881, %r1879, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1882, %r1881, 0f3F800000;
	add.f32 	%r1883, %r1880, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1884, %r1869, %r1883;
	mul.f32 	%r1885, %r1868, %r1882;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r341, %r1885, %r1884;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1886, %r2404, %r470;
	add.f32 	%r1887, %r2400, %r469;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1888, %r1887, 0f3D372713;
	mul.f32 	%r1889, %r1886, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1890, %r1887, %r1888;
	mul.f32 	%r1891, %r1886, %r1889;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1892, %r1887, %r1890, %r1887;
	fma.rn.f32 	%r1893, %r1886, %r1891, %r1886;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1894, %r1892, 0f3F4C422A;
	mul.f32 	%r1895, %r1893, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1896, %r1886, 0f3F000000;
	mul.f32 	%r1897, %r1887, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1898, %r1894, 0fC0000000;
	mul.f32 	%r1899, %r1895, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1900, %r1898, 0f3FB8AA3B;
	ex2.approx.f32 	%r1901, %r1900;
	mul.f32 	%r1902, %r1899, 0f3FB8AA3B;
	ex2.approx.f32 	%r1903, %r1902;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1904, %r1901, 0f3F800000;
	add.f32 	%r1905, %r1903, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1906, %r491, %r1904;
	div.full.f32 	%r1907, %r491, %r1905;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1908, %r1906, 0fBF800000;
	add.f32 	%r1909, %r1907, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1910, %r1909, 0f3F800000;
	add.f32 	%r1911, %r1908, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1912, %r1897, %r1911;
	mul.f32 	%r1913, %r1896, %r1910;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r345, %r1913, %r1912;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1914, %r2405, %r501;
	add.f32 	%r1915, %r2401, %r500;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1916, %r1915, 0f3D372713;
	mul.f32 	%r1917, %r1914, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1918, %r1915, %r1916;
	mul.f32 	%r1919, %r1914, %r1917;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1920, %r1915, %r1918, %r1915;
	fma.rn.f32 	%r1921, %r1914, %r1919, %r1914;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1922, %r1920, 0f3F4C422A;
	mul.f32 	%r1923, %r1921, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1924, %r1914, 0f3F000000;
	mul.f32 	%r1925, %r1915, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1926, %r1922, 0fC0000000;
	mul.f32 	%r1927, %r1923, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1928, %r1926, 0f3FB8AA3B;
	ex2.approx.f32 	%r1929, %r1928;
	mul.f32 	%r1930, %r1927, 0f3FB8AA3B;
	ex2.approx.f32 	%r1931, %r1930;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1932, %r1929, 0f3F800000;
	add.f32 	%r1933, %r1931, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1934, %r491, %r1932;
	div.full.f32 	%r1935, %r491, %r1933;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1936, %r1934, 0fBF800000;
	add.f32 	%r1937, %r1935, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1938, %r1937, 0f3F800000;
	add.f32 	%r1939, %r1936, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1940, %r1925, %r1939;
	mul.f32 	%r1941, %r1924, %r1938;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r349, %r1941, %r1940;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1942, %r2410, %r587;
	add.f32 	%r1943, %r2406, %r586;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1944, %r1943, 0f3D372713;
	mul.f32 	%r1945, %r1942, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1946, %r1943, %r1944;
	mul.f32 	%r1947, %r1942, %r1945;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1948, %r1943, %r1946, %r1943;
	fma.rn.f32 	%r1949, %r1942, %r1947, %r1942;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1950, %r1948, 0f3F4C422A;
	mul.f32 	%r1951, %r1949, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1952, %r1942, 0f3F000000;
	mul.f32 	%r1953, %r1943, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1954, %r1950, 0fC0000000;
	mul.f32 	%r1955, %r1951, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1956, %r1954, 0f3FB8AA3B;
	ex2.approx.f32 	%r1957, %r1956;
	mul.f32 	%r1958, %r1955, 0f3FB8AA3B;
	ex2.approx.f32 	%r1959, %r1958;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1960, %r1957, 0f3F800000;
	add.f32 	%r1961, %r1959, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1962, %r491, %r1960;
	div.full.f32 	%r1963, %r491, %r1961;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1964, %r1962, 0fBF800000;
	add.f32 	%r1965, %r1963, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1966, %r1965, 0f3F800000;
	add.f32 	%r1967, %r1964, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1968, %r1953, %r1967;
	mul.f32 	%r1969, %r1952, %r1966;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r338, %r1969, %r1968;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1970, %r2411, %r617;
	add.f32 	%r1971, %r2407, %r616;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r1972, %r1971, 0f3D372713;
	mul.f32 	%r1973, %r1970, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r1974, %r1971, %r1972;
	mul.f32 	%r1975, %r1970, %r1973;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r1976, %r1971, %r1974, %r1971;
	fma.rn.f32 	%r1977, %r1970, %r1975, %r1970;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r1978, %r1976, 0f3F4C422A;
	mul.f32 	%r1979, %r1977, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r1980, %r1970, 0f3F000000;
	mul.f32 	%r1981, %r1971, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r1982, %r1978, 0fC0000000;
	mul.f32 	%r1983, %r1979, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r1984, %r1982, 0f3FB8AA3B;
	ex2.approx.f32 	%r1985, %r1984;
	mul.f32 	%r1986, %r1983, 0f3FB8AA3B;
	ex2.approx.f32 	%r1987, %r1986;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r1988, %r1985, 0f3F800000;
	add.f32 	%r1989, %r1987, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r1990, %r491, %r1988;
	div.full.f32 	%r1991, %r491, %r1989;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r1992, %r1990, 0fBF800000;
	add.f32 	%r1993, %r1991, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r1994, %r1993, 0f3F800000;
	add.f32 	%r1995, %r1992, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r1996, %r1981, %r1995;
	mul.f32 	%r1997, %r1980, %r1994;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r342, %r1997, %r1996;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r1998, %r2412, %r587;
	add.f32 	%r1999, %r2408, %r586;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r2000, %r1999, 0f3D372713;
	mul.f32 	%r2001, %r1998, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r2002, %r1999, %r2000;
	mul.f32 	%r2003, %r1998, %r2001;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r2004, %r1999, %r2002, %r1999;
	fma.rn.f32 	%r2005, %r1998, %r2003, %r1998;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r2006, %r2004, 0f3F4C422A;
	mul.f32 	%r2007, %r2005, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r2008, %r1998, 0f3F000000;
	mul.f32 	%r2009, %r1999, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r2010, %r2006, 0fC0000000;
	mul.f32 	%r2011, %r2007, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r2012, %r2010, 0f3FB8AA3B;
	ex2.approx.f32 	%r2013, %r2012;
	mul.f32 	%r2014, %r2011, 0f3FB8AA3B;
	ex2.approx.f32 	%r2015, %r2014;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r2016, %r2013, 0f3F800000;
	add.f32 	%r2017, %r2015, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r2018, %r491, %r2016;
	div.full.f32 	%r2019, %r491, %r2017;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r2020, %r2018, 0fBF800000;
	add.f32 	%r2021, %r2019, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r2022, %r2021, 0f3F800000;
	add.f32 	%r2023, %r2020, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r2024, %r2009, %r2023;
	mul.f32 	%r2025, %r2008, %r2022;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r346, %r2025, %r2024;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r2026, %r2413, %r617;
	add.f32 	%r2027, %r2409, %r616;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r2028, %r2027, 0f3D372713;
	mul.f32 	%r2029, %r2026, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r2030, %r2027, %r2028;
	mul.f32 	%r2031, %r2026, %r2029;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r2032, %r2027, %r2030, %r2027;
	fma.rn.f32 	%r2033, %r2026, %r2031, %r2026;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r2034, %r2032, 0f3F4C422A;
	mul.f32 	%r2035, %r2033, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r2036, %r2026, 0f3F000000;
	mul.f32 	%r2037, %r2027, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r2038, %r2034, 0fC0000000;
	mul.f32 	%r2039, %r2035, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r2040, %r2038, 0f3FB8AA3B;
	ex2.approx.f32 	%r2041, %r2040;
	mul.f32 	%r2042, %r2039, 0f3FB8AA3B;
	ex2.approx.f32 	%r2043, %r2042;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r2044, %r2041, 0f3F800000;
	add.f32 	%r2045, %r2043, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r2046, %r491, %r2044;
	div.full.f32 	%r2047, %r491, %r2045;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r2048, %r2046, 0fBF800000;
	add.f32 	%r2049, %r2047, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r2050, %r2049, 0f3F800000;
	add.f32 	%r2051, %r2048, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r2052, %r2037, %r2051;
	mul.f32 	%r2053, %r2036, %r2050;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r350, %r2053, %r2052;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r2054, %r2418, %r703;
	add.f32 	%r2055, %r2414, %r702;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r2056, %r2055, 0f3D372713;
	mul.f32 	%r2057, %r2054, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r2058, %r2055, %r2056;
	mul.f32 	%r2059, %r2054, %r2057;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r2060, %r2055, %r2058, %r2055;
	fma.rn.f32 	%r2061, %r2054, %r2059, %r2054;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r2062, %r2060, 0f3F4C422A;
	mul.f32 	%r2063, %r2061, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r2064, %r2054, 0f3F000000;
	mul.f32 	%r2065, %r2055, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r2066, %r2062, 0fC0000000;
	mul.f32 	%r2067, %r2063, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r2068, %r2066, 0f3FB8AA3B;
	ex2.approx.f32 	%r2069, %r2068;
	mul.f32 	%r2070, %r2067, 0f3FB8AA3B;
	ex2.approx.f32 	%r2071, %r2070;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r2072, %r2069, 0f3F800000;
	add.f32 	%r2073, %r2071, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r2074, %r491, %r2072;
	div.full.f32 	%r2075, %r491, %r2073;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r2076, %r2074, 0fBF800000;
	add.f32 	%r2077, %r2075, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r2078, %r2077, 0f3F800000;
	add.f32 	%r2079, %r2076, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r2080, %r2065, %r2079;
	mul.f32 	%r2081, %r2064, %r2078;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r339, %r2081, %r2080;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r2082, %r2419, %r733;
	add.f32 	%r2083, %r2415, %r732;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r2084, %r2083, 0f3D372713;
	mul.f32 	%r2085, %r2082, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r2086, %r2083, %r2084;
	mul.f32 	%r2087, %r2082, %r2085;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r2088, %r2083, %r2086, %r2083;
	fma.rn.f32 	%r2089, %r2082, %r2087, %r2082;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r2090, %r2088, 0f3F4C422A;
	mul.f32 	%r2091, %r2089, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r2092, %r2082, 0f3F000000;
	mul.f32 	%r2093, %r2083, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r2094, %r2090, 0fC0000000;
	mul.f32 	%r2095, %r2091, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r2096, %r2094, 0f3FB8AA3B;
	ex2.approx.f32 	%r2097, %r2096;
	mul.f32 	%r2098, %r2095, 0f3FB8AA3B;
	ex2.approx.f32 	%r2099, %r2098;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r2100, %r2097, 0f3F800000;
	add.f32 	%r2101, %r2099, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r2102, %r491, %r2100;
	div.full.f32 	%r2103, %r491, %r2101;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r2104, %r2102, 0fBF800000;
	add.f32 	%r2105, %r2103, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r2106, %r2105, 0f3F800000;
	add.f32 	%r2107, %r2104, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r2108, %r2093, %r2107;
	mul.f32 	%r2109, %r2092, %r2106;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r343, %r2109, %r2108;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r2110, %r2420, %r703;
	add.f32 	%r2111, %r2416, %r702;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r2112, %r2111, 0f3D372713;
	mul.f32 	%r2113, %r2110, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r2114, %r2111, %r2112;
	mul.f32 	%r2115, %r2110, %r2113;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r2116, %r2111, %r2114, %r2111;
	fma.rn.f32 	%r2117, %r2110, %r2115, %r2110;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r2118, %r2116, 0f3F4C422A;
	mul.f32 	%r2119, %r2117, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r2120, %r2110, 0f3F000000;
	mul.f32 	%r2121, %r2111, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r2122, %r2118, 0fC0000000;
	mul.f32 	%r2123, %r2119, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r2124, %r2122, 0f3FB8AA3B;
	ex2.approx.f32 	%r2125, %r2124;
	mul.f32 	%r2126, %r2123, 0f3FB8AA3B;
	ex2.approx.f32 	%r2127, %r2126;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r2128, %r2125, 0f3F800000;
	add.f32 	%r2129, %r2127, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r2130, %r491, %r2128;
	div.full.f32 	%r2131, %r491, %r2129;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r2132, %r2130, 0fBF800000;
	add.f32 	%r2133, %r2131, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r2134, %r2133, 0f3F800000;
	add.f32 	%r2135, %r2132, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r2136, %r2121, %r2135;
	mul.f32 	%r2137, %r2120, %r2134;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r347, %r2137, %r2136;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r2138, %r2421, %r733;
	add.f32 	%r2139, %r2417, %r732;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r2140, %r2139, 0f3D372713;
	mul.f32 	%r2141, %r2138, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r2142, %r2139, %r2140;
	mul.f32 	%r2143, %r2138, %r2141;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r2144, %r2139, %r2142, %r2139;
	fma.rn.f32 	%r2145, %r2138, %r2143, %r2138;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r2146, %r2144, 0f3F4C422A;
	mul.f32 	%r2147, %r2145, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r2148, %r2138, 0f3F000000;
	mul.f32 	%r2149, %r2139, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r2150, %r2146, 0fC0000000;
	mul.f32 	%r2151, %r2147, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r2152, %r2150, 0f3FB8AA3B;
	ex2.approx.f32 	%r2153, %r2152;
	mul.f32 	%r2154, %r2151, 0f3FB8AA3B;
	ex2.approx.f32 	%r2155, %r2154;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r2156, %r2153, 0f3F800000;
	add.f32 	%r2157, %r2155, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r2158, %r491, %r2156;
	div.full.f32 	%r2159, %r491, %r2157;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r2160, %r2158, 0fBF800000;
	add.f32 	%r2161, %r2159, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r2162, %r2161, 0f3F800000;
	add.f32 	%r2163, %r2160, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r2164, %r2149, %r2163;
	mul.f32 	%r2165, %r2148, %r2162;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r351, %r2165, %r2164;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r2166, %r2426, %r819;
	add.f32 	%r2167, %r2422, %r818;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r2168, %r2167, 0f3D372713;
	mul.f32 	%r2169, %r2166, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r2170, %r2167, %r2168;
	mul.f32 	%r2171, %r2166, %r2169;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r2172, %r2167, %r2170, %r2167;
	fma.rn.f32 	%r2173, %r2166, %r2171, %r2166;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r2174, %r2172, 0f3F4C422A;
	mul.f32 	%r2175, %r2173, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r2176, %r2166, 0f3F000000;
	mul.f32 	%r2177, %r2167, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r2178, %r2174, 0fC0000000;
	mul.f32 	%r2179, %r2175, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r2180, %r2178, 0f3FB8AA3B;
	ex2.approx.f32 	%r2181, %r2180;
	mul.f32 	%r2182, %r2179, 0f3FB8AA3B;
	ex2.approx.f32 	%r2183, %r2182;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r2184, %r2181, 0f3F800000;
	add.f32 	%r2185, %r2183, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r2186, %r491, %r2184;
	div.full.f32 	%r2187, %r491, %r2185;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r2188, %r2186, 0fBF800000;
	add.f32 	%r2189, %r2187, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r2190, %r2189, 0f3F800000;
	add.f32 	%r2191, %r2188, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r2192, %r2177, %r2191;
	mul.f32 	%r2193, %r2176, %r2190;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r340, %r2193, %r2192;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r2194, %r2427, %r849;
	add.f32 	%r2195, %r2423, %r848;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r2196, %r2195, 0f3D372713;
	mul.f32 	%r2197, %r2194, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r2198, %r2195, %r2196;
	mul.f32 	%r2199, %r2194, %r2197;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r2200, %r2195, %r2198, %r2195;
	fma.rn.f32 	%r2201, %r2194, %r2199, %r2194;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r2202, %r2200, 0f3F4C422A;
	mul.f32 	%r2203, %r2201, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r2204, %r2194, 0f3F000000;
	mul.f32 	%r2205, %r2195, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r2206, %r2202, 0fC0000000;
	mul.f32 	%r2207, %r2203, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r2208, %r2206, 0f3FB8AA3B;
	ex2.approx.f32 	%r2209, %r2208;
	mul.f32 	%r2210, %r2207, 0f3FB8AA3B;
	ex2.approx.f32 	%r2211, %r2210;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r2212, %r2209, 0f3F800000;
	add.f32 	%r2213, %r2211, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r2214, %r491, %r2212;
	div.full.f32 	%r2215, %r491, %r2213;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r2216, %r2214, 0fBF800000;
	add.f32 	%r2217, %r2215, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r2218, %r2217, 0f3F800000;
	add.f32 	%r2219, %r2216, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r2220, %r2205, %r2219;
	mul.f32 	%r2221, %r2204, %r2218;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r344, %r2221, %r2220;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r2222, %r2428, %r819;
	add.f32 	%r2223, %r2424, %r818;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r2224, %r2223, 0f3D372713;
	mul.f32 	%r2225, %r2222, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r2226, %r2223, %r2224;
	mul.f32 	%r2227, %r2222, %r2225;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r2228, %r2223, %r2226, %r2223;
	fma.rn.f32 	%r2229, %r2222, %r2227, %r2222;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r2230, %r2228, 0f3F4C422A;
	mul.f32 	%r2231, %r2229, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r2232, %r2222, 0f3F000000;
	mul.f32 	%r2233, %r2223, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r2234, %r2230, 0fC0000000;
	mul.f32 	%r2235, %r2231, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r2236, %r2234, 0f3FB8AA3B;
	ex2.approx.f32 	%r2237, %r2236;
	mul.f32 	%r2238, %r2235, 0f3FB8AA3B;
	ex2.approx.f32 	%r2239, %r2238;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r2240, %r2237, 0f3F800000;
	add.f32 	%r2241, %r2239, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r2242, %r491, %r2240;
	div.full.f32 	%r2243, %r491, %r2241;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r2244, %r2242, 0fBF800000;
	add.f32 	%r2245, %r2243, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r2246, %r2245, 0f3F800000;
	add.f32 	%r2247, %r2244, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r2248, %r2233, %r2247;
	mul.f32 	%r2249, %r2232, %r2246;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r348, %r2249, %r2248;
	.loc	1 95 14                         // sk11_vision.py:95:14
	add.f32 	%r2250, %r2429, %r849;
	add.f32 	%r2251, %r2425, %r848;
	.loc	1 96 49                         // sk11_vision.py:96:49
	mul.f32 	%r2252, %r2251, 0f3D372713;
	mul.f32 	%r2253, %r2250, 0f3D372713;
	.loc	1 96 53                         // sk11_vision.py:96:53
	mul.f32 	%r2254, %r2251, %r2252;
	mul.f32 	%r2255, %r2250, %r2253;
	.loc	1 96 38                         // sk11_vision.py:96:38
	fma.rn.f32 	%r2256, %r2251, %r2254, %r2251;
	fma.rn.f32 	%r2257, %r2250, %r2255, %r2250;
	.loc	1 96 34                         // sk11_vision.py:96:34
	mul.f32 	%r2258, %r2256, 0f3F4C422A;
	mul.f32 	%r2259, %r2257, 0f3F4C422A;
	.loc	1 97 14                         // sk11_vision.py:97:14
	mul.f32 	%r2260, %r2250, 0f3F000000;
	mul.f32 	%r2261, %r2251, 0f3F000000;
	.loc	1 97 53                         // sk11_vision.py:97:53
	mul.f32 	%r2262, %r2258, 0fC0000000;
	mul.f32 	%r2263, %r2259, 0fC0000000;
	.loc	1 97 46                         // sk11_vision.py:97:46
	mul.f32 	%r2264, %r2262, 0f3FB8AA3B;
	ex2.approx.f32 	%r2265, %r2264;
	mul.f32 	%r2266, %r2263, 0f3FB8AA3B;
	ex2.approx.f32 	%r2267, %r2266;
	.loc	1 97 39                         // sk11_vision.py:97:39
	add.f32 	%r2268, %r2265, 0f3F800000;
	add.f32 	%r2269, %r2267, 0f3F800000;
	.loc	1 97 33                         // sk11_vision.py:97:33
	div.full.f32 	%r2270, %r491, %r2268;
	div.full.f32 	%r2271, %r491, %r2269;
	.loc	1 97 63                         // sk11_vision.py:97:63
	add.f32 	%r2272, %r2270, 0fBF800000;
	add.f32 	%r2273, %r2271, 0fBF800000;
	.loc	1 97 26                         // sk11_vision.py:97:26
	add.f32 	%r2274, %r2273, 0f3F800000;
	add.f32 	%r2275, %r2272, 0f3F800000;
	.loc	1 97 19                         // sk11_vision.py:97:19
	mul.f32 	%r2276, %r2261, %r2275;
	mul.f32 	%r2277, %r2260, %r2274;
	.loc	1 103 13                        // sk11_vision.py:103:13
	cvt.rn.bf16x2.f32 	%r352, %r2277, %r2276;
	.loc	1 103 8                         // sk11_vision.py:103:8
	bar.sync 	0;
	shl.b32 	%r2278, %r448, 5;
	and.b32 	%r2279, %r2, 24;
	shl.b32 	%r2280, %r2279, 4;
	bfe.s32 	%r2281, %r2, 2, 1;
	shl.b32 	%r2282, %r2, 2;
	or.b32 	%r2283, %r2278, %r2280;
	and.b32 	%r2284, %r2281, 1040;
	and.b32 	%r2285, %r2282, 512;
	xor.b32 	%r2286, %r2283, %r450;
	shl.b32 	%r2287, %r448, 11;
	or.b32 	%r2288, %r2285, %r2284;
	or.b32 	%r2289, %r2286, %r2287;
	or.b32 	%r2290, %r2289, %r2288;
	add.s32 	%r287, %r119, %r2290;
	// begin inline asm
	st.shared.v4.b32 [ %r287 + 0 ], { %r288, %r289, %r290, %r291 };
	// end inline asm
	xor.b32 	%r2291, %r2290, 16;
	add.s32 	%r292, %r119, %r2291;
	// begin inline asm
	st.shared.v4.b32 [ %r292 + 0 ], { %r293, %r294, %r295, %r296 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r2292, %r2279, 8;
	shl.b32 	%r2293, %r2279, 2;
	and.b32 	%r2294, %r443, 1040;
	or.b32 	%r2295, %r2292, %r2431;
	or.b32 	%r2296, %r2293, %r442;
	xor.b32 	%r2297, %r2295, %r2296;
	xor.b32 	%r2298, %r2297, %r2294;
	add.s32 	%r2299, %r119, %r2298;
	ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r353, %r354, %r355, %r356}, [%r2299];
	ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r361, %r362, %r363, %r364}, [%r2299+512];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r287 + 0 ], { %r297, %r298, %r299, %r300 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r292 + 0 ], { %r301, %r302, %r303, %r304 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r357, %r358, %r359, %r360}, [%r2299];
	ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r365, %r366, %r367, %r368}, [%r2299+512];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r287 + 0 ], { %r305, %r306, %r307, %r308 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r292 + 0 ], { %r309, %r310, %r311, %r312 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r369, %r370, %r371, %r372}, [%r2299];
	ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r377, %r378, %r379, %r380}, [%r2299+512];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r287 + 0 ], { %r313, %r314, %r315, %r316 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r292 + 0 ], { %r317, %r318, %r319, %r320 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r373, %r374, %r375, %r376}, [%r2299];
	ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r381, %r382, %r383, %r384}, [%r2299+512];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r287 + 0 ], { %r321, %r322, %r323, %r324 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r292 + 0 ], { %r325, %r326, %r327, %r328 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r385, %r386, %r387, %r388}, [%r2299];
	ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r393, %r394, %r395, %r396}, [%r2299+512];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r287 + 0 ], { %r329, %r330, %r331, %r332 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r292 + 0 ], { %r333, %r334, %r335, %r336 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r389, %r390, %r391, %r392}, [%r2299];
	ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r397, %r398, %r399, %r400}, [%r2299+512];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r287 + 0 ], { %r337, %r338, %r339, %r340 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r292 + 0 ], { %r341, %r342, %r343, %r344 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r401, %r402, %r403, %r404}, [%r2299];
	ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r409, %r410, %r411, %r412}, [%r2299+512];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r287 + 0 ], { %r345, %r346, %r347, %r348 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r292 + 0 ], { %r349, %r350, %r351, %r352 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r405, %r406, %r407, %r408}, [%r2299];
	ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r413, %r414, %r415, %r416}, [%r2299+512];
	// begin inline asm
	@%p7 st.global.v4.b32 [ %rd85 + 0 ], { %r353, %r354, %r355, %r356 };
	// end inline asm
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd86 + 0 ], { %r357, %r358, %r359, %r360 };
	// end inline asm
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd87 + 0 ], { %r361, %r362, %r363, %r364 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd88 + 0 ], { %r365, %r366, %r367, %r368 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd89 + 0 ], { %r369, %r370, %r371, %r372 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd90 + 0 ], { %r373, %r374, %r375, %r376 };
	// end inline asm
	// begin inline asm
	@%p13 st.global.v4.b32 [ %rd91 + 0 ], { %r377, %r378, %r379, %r380 };
	// end inline asm
	// begin inline asm
	@%p14 st.global.v4.b32 [ %rd92 + 0 ], { %r381, %r382, %r383, %r384 };
	// end inline asm
	// begin inline asm
	@%p15 st.global.v4.b32 [ %rd93 + 0 ], { %r385, %r386, %r387, %r388 };
	// end inline asm
	// begin inline asm
	@%p16 st.global.v4.b32 [ %rd94 + 0 ], { %r389, %r390, %r391, %r392 };
	// end inline asm
	// begin inline asm
	@%p17 st.global.v4.b32 [ %rd95 + 0 ], { %r393, %r394, %r395, %r396 };
	// end inline asm
	// begin inline asm
	@%p18 st.global.v4.b32 [ %rd96 + 0 ], { %r397, %r398, %r399, %r400 };
	// end inline asm
	// begin inline asm
	@%p19 st.global.v4.b32 [ %rd97 + 0 ], { %r401, %r402, %r403, %r404 };
	// end inline asm
	// begin inline asm
	@%p20 st.global.v4.b32 [ %rd98 + 0 ], { %r405, %r406, %r407, %r408 };
	// end inline asm
	// begin inline asm
	@%p21 st.global.v4.b32 [ %rd99 + 0 ], { %r409, %r410, %r411, %r412 };
	// end inline asm
	// begin inline asm
	@%p22 st.global.v4.b32 [ %rd100 + 0 ], { %r413, %r414, %r415, %r416 };
	// end inline asm
	.loc	1 101 4                         // sk11_vision.py:101:4
	ret;
$L__tmp6:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk11_vision.py"
	.file	2 "/usr/local/lib/python3.12/dist-packages/triton/language/standard.py"
	.section	.debug_abbrev
	{
.b8 1                                   // Abbreviation Code
.b8 17                                  // DW_TAG_compile_unit
.b8 1                                   // DW_CHILDREN_yes
.b8 37                                  // DW_AT_producer
.b8 8                                   // DW_FORM_string
.b8 19                                  // DW_AT_language
.b8 5                                   // DW_FORM_data2
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 16                                  // DW_AT_stmt_list
.b8 6                                   // DW_FORM_data4
.b8 27                                  // DW_AT_comp_dir
.b8 8                                   // DW_FORM_string
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 2                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 0                                   // DW_CHILDREN_no
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 32                                  // DW_AT_inline
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 3                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 1                                   // DW_CHILDREN_yes
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 4                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 0                                   // DW_CHILDREN_no
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 11                                  // DW_FORM_data1
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 186                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xb3 DW_TAG_compile_unit
.b8 116                                 // DW_AT_producer
.b8 114
.b8 105
.b8 116
.b8 111
.b8 110
.b8 0
.b8 2                                   // DW_AT_language
.b8 0
.b8 115                                 // DW_AT_name
.b8 107
.b8 49
.b8 49
.b8 95
.b8 118
.b8 105
.b8 115
.b8 105
.b8 111
.b8 110
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 114
.b8 101
.b8 112
.b8 111
.b8 47
.b8 118
.b8 108
.b8 108
.b8 109
.b8 47
.b8 95
.b8 103
.b8 101
.b8 110
.b8 101
.b8 115
.b8 105
.b8 115
.b8 47
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 115
.b8 0
.b8 2                                   // Abbrev [2] 0x44:0x1b DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 49
.b8 49
.b8 95
.b8 108
.b8 105
.b8 110
.b8 101
.b8 97
.b8 114
.b8 95
.b8 103
.b8 101
.b8 108
.b8 117
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x5f:0x5e DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 68                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x74:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 74                                  // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x8c:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 75                                  // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0xa4:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp4                           // DW_AT_low_pc
.b64 $L__tmp5                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 90                                  // DW_AT_call_line
.b8 33                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_2 = _Nativo(
    "sk11_vision/tile128x256x64_shift0_abi10",
    _PTX_2, "_sk11_linear_gelu_kernel",
    warps=8, shared=98304,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 10, 11],
    horneado={8: 1, 9: 1, 12: 1, 13: 1, 14: 128, 15: 256, 16: 64, 17: 8},
    div16=[5, 6, 7, 10, 11],
)


# (cfg, has_shift) -> variantes candidatas. La eleccion final
# entre ellas la hace _lanzar comparando los valores horneados:
# con residual y sin residual el ABI tiene distinta cantidad de
# params, porque stride_res_n solo se especializa cuando vale 1.
_POR_CFG = {}
_POR_CFG.setdefault(((16, 128, 64, 8, 4, 4), False), []).append(_VAR_0)
_POR_CFG.setdefault(((64, 128, 64, 8, 8, 4), False), []).append(_VAR_1)
_POR_CFG.setdefault(((128, 256, 64, 8, 8, 3), False), []).append(_VAR_2)


def _lanzar(grid, cfg, has_shift, *args):
    """Elige la variante cuyos valores horneados coinciden con estos args."""
    cands = _POR_CFG.get((cfg, bool(has_shift)))
    if not cands:
        raise KeyError("no hay PTX embebido para cfg=%r has_shift=%r" % (cfg, has_shift))
    for v in cands:
        if all(args[p] == val for p, val in v.horneado.items()):
            return v((grid,) if isinstance(grid, int) else grid, *args)
    raise KeyError("no hay variante para cfg=%r has_shift=%r con estos strides" % (cfg, has_shift))


# --- quant: PTX embebido (E1=0 lop3, E2=0 mul) ---

_QPTX0 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk11_bias_gelu_kernel  // -- Begin function _sk11_bias_gelu_kernel
                                        // @_sk11_bias_gelu_kernel
.visible .entry _sk11_bias_gelu_kernel(
	.param .u64 .ptr .global .align 1 _sk11_bias_gelu_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk11_bias_gelu_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk11_bias_gelu_kernel_param_2,
	.param .u32 _sk11_bias_gelu_kernel_param_3,
	.param .u32 _sk11_bias_gelu_kernel_param_4,
	.param .u32 _sk11_bias_gelu_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk11_bias_gelu_kernel_param_6,
	.param .u64 .ptr .global .align 1 _sk11_bias_gelu_kernel_param_7
)
.reqntid 256
{
	.reg .pred 	%p<2>;
	.reg .b16 	%rs<17>;
	.reg .b32 	%r<152>;
	.reg .b64 	%rd<10>;
	.loc	1 109 0                         // sk11_vision.py:109:0
$L__func_begin0:
	.loc	1 109 0                         // sk11_vision.py:109:0

// %bb.0:
	ld.param.b64 	%rd4, [_sk11_bias_gelu_kernel_param_0];
	ld.param.b64 	%rd5, [_sk11_bias_gelu_kernel_param_1];
$L__tmp0:
	.loc	1 111 24                        // sk11_vision.py:111:24
	mov.u32 	%r14, %ctaid.x;
	ld.param.b64 	%rd6, [_sk11_bias_gelu_kernel_param_2];
	.loc	1 112 24                        // sk11_vision.py:112:24
	mov.u32 	%r15, %tid.x;
	shl.b32 	%r16, %r15, 3;
	ld.param.b32 	%r17, [_sk11_bias_gelu_kernel_param_3];
	and.b32 	%r18, %r16, 2040;
	ld.param.b32 	%r19, [_sk11_bias_gelu_kernel_param_4];
	.loc	1 113 18                        // sk11_vision.py:113:18
	setp.lt.s32 	%p1, %r18, %r17;
	ld.param.b32 	%r20, [_sk11_bias_gelu_kernel_param_5];
	.loc	1 114 30                        // sk11_vision.py:114:30
	mul.lo.s32 	%r21, %r19, %r14;
	.loc	1 114 24                        // sk11_vision.py:114:24
	mad.wide.s32 	%rd7, %r21, 2, %rd4;
	.loc	1 114 42                        // sk11_vision.py:114:42
	mul.wide.u32 	%rd8, %r18, 2;
	add.s64 	%rd1, %rd7, %rd8;
	mov.b32 	%r5, 0;
	.loc	1 114 16                        // sk11_vision.py:114:16
	// begin inline asm
	mov.u32 %r1, %r5;
	mov.u32 %r2, %r5;
	mov.u32 %r3, %r5;
	mov.u32 %r4, %r5;
	@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
	// end inline asm
	.loc	1 115 28                        // sk11_vision.py:115:28
	add.s64 	%rd2, %rd5, %rd8;
	.loc	1 115 17                        // sk11_vision.py:115:17
	// begin inline asm
	mov.u32 %r6, %r5;
	mov.u32 %r7, %r5;
	mov.u32 %r8, %r5;
	mov.u32 %r9, %r5;
	@%p1 ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd2 + 0 ];
	// end inline asm
	.loc	1 118 29                        // sk11_vision.py:118:29
	mul.lo.s32 	%r22, %r20, %r14;
	.loc	1 118 23                        // sk11_vision.py:118:23
	mad.wide.s32 	%rd9, %r22, 2, %rd6;
	.loc	1 118 41                        // sk11_vision.py:118:41
	add.s64 	%rd3, %rd9, %rd8;
	.loc	1 114 73                        // sk11_vision.py:114:73
	mov.b32 	{%rs1, %rs2}, %r1;
	cvt.f32.bf16 	%r23, %rs1;
	cvt.f32.bf16 	%r24, %rs2;
	.loc	1 115 59                        // sk11_vision.py:115:59
	mov.b32 	{%rs3, %rs4}, %r6;
	cvt.f32.bf16 	%r25, %rs3;
	cvt.f32.bf16 	%r26, %rs4;
	.loc	1 115 9                         // sk11_vision.py:115:9
	add.f32 	%r27, %r24, %r26;
	add.f32 	%r28, %r23, %r25;
	.loc	1 116 49                        // sk11_vision.py:116:49
	mul.f32 	%r29, %r28, 0f3D372713;
	mul.f32 	%r30, %r27, 0f3D372713;
	.loc	1 116 53                        // sk11_vision.py:116:53
	mul.f32 	%r31, %r28, %r29;
	mul.f32 	%r32, %r27, %r30;
	.loc	1 116 38                        // sk11_vision.py:116:38
	fma.rn.f32 	%r33, %r28, %r31, %r28;
	fma.rn.f32 	%r34, %r27, %r32, %r27;
	.loc	1 116 34                        // sk11_vision.py:116:34
	mul.f32 	%r35, %r33, 0f3F4C422A;
	mul.f32 	%r36, %r34, 0f3F4C422A;
	.loc	1 117 14                        // sk11_vision.py:117:14
	mul.f32 	%r37, %r27, 0f3F000000;
	mul.f32 	%r38, %r28, 0f3F000000;
	.loc	1 117 53                        // sk11_vision.py:117:53
	mul.f32 	%r39, %r35, 0fC0000000;
	mul.f32 	%r40, %r36, 0fC0000000;
	.loc	1 117 46                        // sk11_vision.py:117:46
	mul.f32 	%r41, %r39, 0f3FB8AA3B;
	ex2.approx.f32 	%r42, %r41;
	mul.f32 	%r43, %r40, 0f3FB8AA3B;
	ex2.approx.f32 	%r44, %r43;
	.loc	1 117 39                        // sk11_vision.py:117:39
	add.f32 	%r45, %r42, 0f3F800000;
	add.f32 	%r46, %r44, 0f3F800000;
	mov.b32 	%r47, 0f40000000;
	.loc	1 117 33                        // sk11_vision.py:117:33
	div.full.f32 	%r48, %r47, %r45;
	div.full.f32 	%r49, %r47, %r46;
	.loc	1 117 63                        // sk11_vision.py:117:63
	add.f32 	%r50, %r48, 0fBF800000;
	add.f32 	%r51, %r49, 0fBF800000;
	.loc	1 117 26                        // sk11_vision.py:117:26
	add.f32 	%r52, %r51, 0f3F800000;
	add.f32 	%r53, %r50, 0f3F800000;
	.loc	1 117 19                        // sk11_vision.py:117:19
	mul.f32 	%r54, %r38, %r53;
	mul.f32 	%r55, %r37, %r52;
	.loc	1 118 52                        // sk11_vision.py:118:52
	cvt.rn.bf16x2.f32 	%r10, %r55, %r54;
	.loc	1 114 73                        // sk11_vision.py:114:73
	mov.b32 	{%rs5, %rs6}, %r2;
	cvt.f32.bf16 	%r56, %rs5;
	cvt.f32.bf16 	%r57, %rs6;
	.loc	1 115 59                        // sk11_vision.py:115:59
	mov.b32 	{%rs7, %rs8}, %r7;
	cvt.f32.bf16 	%r58, %rs7;
	cvt.f32.bf16 	%r59, %rs8;
	.loc	1 115 9                         // sk11_vision.py:115:9
	add.f32 	%r60, %r57, %r59;
	add.f32 	%r61, %r56, %r58;
	.loc	1 116 49                        // sk11_vision.py:116:49
	mul.f32 	%r62, %r61, 0f3D372713;
	mul.f32 	%r63, %r60, 0f3D372713;
	.loc	1 116 53                        // sk11_vision.py:116:53
	mul.f32 	%r64, %r61, %r62;
	mul.f32 	%r65, %r60, %r63;
	.loc	1 116 38                        // sk11_vision.py:116:38
	fma.rn.f32 	%r66, %r61, %r64, %r61;
	fma.rn.f32 	%r67, %r60, %r65, %r60;
	.loc	1 116 34                        // sk11_vision.py:116:34
	mul.f32 	%r68, %r66, 0f3F4C422A;
	mul.f32 	%r69, %r67, 0f3F4C422A;
	.loc	1 117 14                        // sk11_vision.py:117:14
	mul.f32 	%r70, %r60, 0f3F000000;
	mul.f32 	%r71, %r61, 0f3F000000;
	.loc	1 117 53                        // sk11_vision.py:117:53
	mul.f32 	%r72, %r68, 0fC0000000;
	mul.f32 	%r73, %r69, 0fC0000000;
	.loc	1 117 46                        // sk11_vision.py:117:46
	mul.f32 	%r74, %r72, 0f3FB8AA3B;
	ex2.approx.f32 	%r75, %r74;
	mul.f32 	%r76, %r73, 0f3FB8AA3B;
	ex2.approx.f32 	%r77, %r76;
	.loc	1 117 39                        // sk11_vision.py:117:39
	add.f32 	%r78, %r75, 0f3F800000;
	add.f32 	%r79, %r77, 0f3F800000;
	.loc	1 117 33                        // sk11_vision.py:117:33
	div.full.f32 	%r80, %r47, %r78;
	div.full.f32 	%r81, %r47, %r79;
	.loc	1 117 63                        // sk11_vision.py:117:63
	add.f32 	%r82, %r80, 0fBF800000;
	add.f32 	%r83, %r81, 0fBF800000;
	.loc	1 117 26                        // sk11_vision.py:117:26
	add.f32 	%r84, %r83, 0f3F800000;
	add.f32 	%r85, %r82, 0f3F800000;
	.loc	1 117 19                        // sk11_vision.py:117:19
	mul.f32 	%r86, %r71, %r85;
	mul.f32 	%r87, %r70, %r84;
	.loc	1 118 52                        // sk11_vision.py:118:52
	cvt.rn.bf16x2.f32 	%r11, %r87, %r86;
	.loc	1 114 73                        // sk11_vision.py:114:73
	mov.b32 	{%rs9, %rs10}, %r3;
	cvt.f32.bf16 	%r88, %rs9;
	cvt.f32.bf16 	%r89, %rs10;
	.loc	1 115 59                        // sk11_vision.py:115:59
	mov.b32 	{%rs11, %rs12}, %r8;
	cvt.f32.bf16 	%r90, %rs11;
	cvt.f32.bf16 	%r91, %rs12;
	.loc	1 115 9                         // sk11_vision.py:115:9
	add.f32 	%r92, %r89, %r91;
	add.f32 	%r93, %r88, %r90;
	.loc	1 116 49                        // sk11_vision.py:116:49
	mul.f32 	%r94, %r93, 0f3D372713;
	mul.f32 	%r95, %r92, 0f3D372713;
	.loc	1 116 53                        // sk11_vision.py:116:53
	mul.f32 	%r96, %r93, %r94;
	mul.f32 	%r97, %r92, %r95;
	.loc	1 116 38                        // sk11_vision.py:116:38
	fma.rn.f32 	%r98, %r93, %r96, %r93;
	fma.rn.f32 	%r99, %r92, %r97, %r92;
	.loc	1 116 34                        // sk11_vision.py:116:34
	mul.f32 	%r100, %r98, 0f3F4C422A;
	mul.f32 	%r101, %r99, 0f3F4C422A;
	.loc	1 117 14                        // sk11_vision.py:117:14
	mul.f32 	%r102, %r92, 0f3F000000;
	mul.f32 	%r103, %r93, 0f3F000000;
	.loc	1 117 53                        // sk11_vision.py:117:53
	mul.f32 	%r104, %r100, 0fC0000000;
	mul.f32 	%r105, %r101, 0fC0000000;
	.loc	1 117 46                        // sk11_vision.py:117:46
	mul.f32 	%r106, %r104, 0f3FB8AA3B;
	ex2.approx.f32 	%r107, %r106;
	mul.f32 	%r108, %r105, 0f3FB8AA3B;
	ex2.approx.f32 	%r109, %r108;
	.loc	1 117 39                        // sk11_vision.py:117:39
	add.f32 	%r110, %r107, 0f3F800000;
	add.f32 	%r111, %r109, 0f3F800000;
	.loc	1 117 33                        // sk11_vision.py:117:33
	div.full.f32 	%r112, %r47, %r110;
	div.full.f32 	%r113, %r47, %r111;
	.loc	1 117 63                        // sk11_vision.py:117:63
	add.f32 	%r114, %r112, 0fBF800000;
	add.f32 	%r115, %r113, 0fBF800000;
	.loc	1 117 26                        // sk11_vision.py:117:26
	add.f32 	%r116, %r115, 0f3F800000;
	add.f32 	%r117, %r114, 0f3F800000;
	.loc	1 117 19                        // sk11_vision.py:117:19
	mul.f32 	%r118, %r103, %r117;
	mul.f32 	%r119, %r102, %r116;
	.loc	1 118 52                        // sk11_vision.py:118:52
	cvt.rn.bf16x2.f32 	%r12, %r119, %r118;
	.loc	1 114 73                        // sk11_vision.py:114:73
	mov.b32 	{%rs13, %rs14}, %r4;
	cvt.f32.bf16 	%r120, %rs13;
	cvt.f32.bf16 	%r121, %rs14;
	.loc	1 115 59                        // sk11_vision.py:115:59
	mov.b32 	{%rs15, %rs16}, %r9;
	cvt.f32.bf16 	%r122, %rs15;
	cvt.f32.bf16 	%r123, %rs16;
	.loc	1 115 9                         // sk11_vision.py:115:9
	add.f32 	%r124, %r121, %r123;
	add.f32 	%r125, %r120, %r122;
	.loc	1 116 49                        // sk11_vision.py:116:49
	mul.f32 	%r126, %r125, 0f3D372713;
	mul.f32 	%r127, %r124, 0f3D372713;
	.loc	1 116 53                        // sk11_vision.py:116:53
	mul.f32 	%r128, %r125, %r126;
	mul.f32 	%r129, %r124, %r127;
	.loc	1 116 38                        // sk11_vision.py:116:38
	fma.rn.f32 	%r130, %r125, %r128, %r125;
	fma.rn.f32 	%r131, %r124, %r129, %r124;
	.loc	1 116 34                        // sk11_vision.py:116:34
	mul.f32 	%r132, %r130, 0f3F4C422A;
	mul.f32 	%r133, %r131, 0f3F4C422A;
	.loc	1 117 14                        // sk11_vision.py:117:14
	mul.f32 	%r134, %r124, 0f3F000000;
	mul.f32 	%r135, %r125, 0f3F000000;
	.loc	1 117 53                        // sk11_vision.py:117:53
	mul.f32 	%r136, %r132, 0fC0000000;
	mul.f32 	%r137, %r133, 0fC0000000;
	.loc	1 117 46                        // sk11_vision.py:117:46
	mul.f32 	%r138, %r136, 0f3FB8AA3B;
	ex2.approx.f32 	%r139, %r138;
	mul.f32 	%r140, %r137, 0f3FB8AA3B;
	ex2.approx.f32 	%r141, %r140;
	.loc	1 117 39                        // sk11_vision.py:117:39
	add.f32 	%r142, %r139, 0f3F800000;
	add.f32 	%r143, %r141, 0f3F800000;
	.loc	1 117 33                        // sk11_vision.py:117:33
	div.full.f32 	%r144, %r47, %r142;
	div.full.f32 	%r145, %r47, %r143;
	.loc	1 117 63                        // sk11_vision.py:117:63
	add.f32 	%r146, %r144, 0fBF800000;
	add.f32 	%r147, %r145, 0fBF800000;
	.loc	1 117 26                        // sk11_vision.py:117:26
	add.f32 	%r148, %r147, 0f3F800000;
	add.f32 	%r149, %r146, 0f3F800000;
	.loc	1 117 19                        // sk11_vision.py:117:19
	mul.f32 	%r150, %r135, %r149;
	mul.f32 	%r151, %r134, %r148;
	.loc	1 118 52                        // sk11_vision.py:118:52
	cvt.rn.bf16x2.f32 	%r13, %r151, %r150;
	.loc	1 118 47                        // sk11_vision.py:118:47
	// begin inline asm
	@%p1 st.global.v4.b32 [ %rd3 + 0 ], { %r10, %r11, %r12, %r13 };
	// end inline asm
	.loc	1 118 4                         // sk11_vision.py:118:4
	ret;
$L__tmp1:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk11_vision.py"
	.section	.debug_abbrev
	{
.b8 1                                   // Abbreviation Code
.b8 17                                  // DW_TAG_compile_unit
.b8 0                                   // DW_CHILDREN_no
.b8 37                                  // DW_AT_producer
.b8 8                                   // DW_FORM_string
.b8 19                                  // DW_AT_language
.b8 5                                   // DW_FORM_data2
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 16                                  // DW_AT_stmt_list
.b8 6                                   // DW_FORM_data4
.b8 27                                  // DW_AT_comp_dir
.b8 8                                   // DW_FORM_string
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 64                                 // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x39 DW_TAG_compile_unit
.b8 116                                 // DW_AT_producer
.b8 114
.b8 105
.b8 116
.b8 111
.b8 110
.b8 0
.b8 2                                   // DW_AT_language
.b8 0
.b8 115                                 // DW_AT_name
.b8 107
.b8 49
.b8 49
.b8 95
.b8 118
.b8 105
.b8 115
.b8 105
.b8 111
.b8 110
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 114
.b8 101
.b8 112
.b8 111
.b8 47
.b8 118
.b8 108
.b8 108
.b8 109
.b8 47
.b8 95
.b8 103
.b8 101
.b8 110
.b8 101
.b8 115
.b8 105
.b8 115
.b8 47
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 115
.b8 0
	}
	.section	.debug_macinfo	{	}
"""

_QVAR0 = _Nativo(
    "sk11_vision/_sk11_bias_gelu_kernel",
    _QPTX0, "_sk11_bias_gelu_kernel",
    warps=8, shared=0,
    abi=[0, 1, 2, 3, 4, 5],
    horneado={6: 2048},
    div16=[3, 4, 5],
)


def _q0_impl(grid: list[int], x_ptr: torch.Tensor, bias_ptr: torch.Tensor, out_ptr: torch.Tensor, N: int, stride_xm: int, stride_om: int, BLOCK: int) -> None:
    """Cuerpo del custom op: lanza el PTX embebido."""
    _QVAR0(tuple(grid), x_ptr, bias_ptr, out_ptr, N, stride_xm, stride_om, BLOCK)


def _q0_fake(grid: list[int], x_ptr: torch.Tensor, bias_ptr: torch.Tensor, out_ptr: torch.Tensor, N: int, stride_xm: int, stride_om: int, BLOCK: int) -> None:
    """Meta impl: no toca la GPU; las salidas se mutan in-place."""
    return None


_q0_registrado = False


def _q0_registrar() -> None:
    """Registra el op una sola vez, fuera de la region compilada."""
    global _q0_registrado
    if _q0_registrado:
        return
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(
        op_name="genesis_sk11_vision_q0",
        op_func=_q0_impl,
        mutates_args=['out_ptr'],
        fake_impl=_q0_fake,
    )
    _q0_registrado = True


def _lanzar_quant0(grid, *args):
    """PTX embebido via custom op; cae al Triton con el kill-switch.

    ``GENESIS_PTQ_NATIVO=0`` vuelve al kernel Triton, que queda en este
    archivo como referencia para los tests.
    """
    if habilitado():
        _q0_registrar()
        g = [grid] if isinstance(grid, int) else list(grid)
        return torch.ops.vllm.genesis_sk11_vision_q0(g, *args)
    ce = ['BLOCK']
    nom = ['x_ptr', 'bias_ptr', 'out_ptr', 'N', 'stride_xm', 'stride_om', 'BLOCK']
    return _sk11_bias_gelu_kernel[grid](*args[:len(nom) - len(ce)],
                    **{n: args[nom.index(n)] for n in ce},
                    num_warps=8, num_stages=1)
