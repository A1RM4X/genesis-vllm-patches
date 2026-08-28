# SPDX-License-Identifier: Apache-2.0
"""SK-08 — SSM_CONTROL — capa completa del control GDN (decode) en GPU.

Capa completa (Qwen3.5-27B, 48 capas GDN):
    mixed_qkv bf16 [B, 10240]      (salida de in_proj_qkv, ya con conv pendiente)
      -> conv1d causal depthwise width=4 + SiLU + shift de estado   (kernel 1)
      -> gated delta rule recurrente + escritura de ssm_state       (kernel 2)
      -> o [B, HV, V]

Geometría: H=16 cabezas K, HV=48 cabezas V, K=V=128,
``conv_dim = 2*H*K + HV*V = 2048 + 2048 + 6144 = 10240``.
``ssm_state [B, HV, V, K]``, ``conv_state [B, conv_dim, 3]``.

Corrección respecto de la versión anterior
------------------------------------------
La versión anterior **no implementaba la delta rule**. Hacía:

    hk_sum   = sum(b_h, axis=1) * 0.01
    b_v_corr = (v - hk_sum) * beta
    b_h     += b_v_corr[:, None] * 0.5
    o        = sum(b_h, axis=1) + q * 0.1

con constantes ``0.01``/``0.5``/``0.1`` sin origen, **sin el vector k en
ninguna parte** (no había producto externo ``k (x) v`` para actualizar el
estado ni proyección por ``q`` para la salida: usaba ``sum(b_h, axis=1)``, que
es sumar el estado, no proyectarlo), y con ``q = v = conv_acc``, es decir q, k
y v eran el mismo tensor. Producía números, no la recurrencia.

La recurrencia correcta, espejo de
``vllm/model_executor/layers/fla/ops/fused_recurrent.py``:

    b_q, b_k  <- L2-normalizados;  b_q <- b_q * scale
    g         = -exp(A_log) * softplus(a + dt_bias)
    beta      = sigmoid(b)
    S        *= exp(g)                       # decaimiento
    b_v      -= sum(S * k[None, :], axis=1)   # error de predicción: v - S k
    b_v      *= beta
    S        += b_v[:, None] * k[None, :]     # producto externo
    o         = sum(S * q[None, :], axis=1)   # proyección por q

El mapeo de cabezas también estaba mal: ``i_h = i_hv // (HV // H)`` (3 cabezas
V por cabeza K), y los offsets ``q_off = i_h*K``, ``k_off = H*K + i_h*K``,
``v_off = 2*H*K + i_hv*V`` sobre ``mixed_qkv``.

Por qué dos kernels y no uno
----------------------------
Los canales de q y k los comparten 3 cabezas V. Si el kernel de la recurrencia
hiciera también el shift del estado del conv, esas 3 cabezas escribirían el
mismo canal mientras las otras aún lo leen: carrera de lectura-escritura, con
resultado dependiente del orden de los CTA. Los canales de v sí son exclusivos
por cabeza, pero separar el conv completo es lo único correcto sin sincronizar
entre bloques. El conv es bandwidth-bound y trivialmente paralelo.

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



SK_ID = "SK-08"
SK_NAME = "SSM_CONTROL"
SK08_CONV_DIM: int = 10240
SK08_CONV_KERNEL_WIDTH: int = 4
SK08_NUM_GDN_LAYERS: int = 48
SK08_NUM_K_HEADS: int = 16
SK08_NUM_V_HEADS: int = 48
SK08_HEAD_DIM: int = 128
SK08_CONV1D_SHAPE: tuple[int, int, int] = (10240, 1, 4)
SK08_A_LOG_SHAPE: tuple[int, ...] = (48,)
SK08_DT_BIAS_SHAPE: tuple[int, ...] = (48,)
SOFTPLUS_THRESHOLD: float = 20.0


@triton.jit
def _sk08_conv1d_silu_kernel(
    x_ptr, w_ptr, bias_ptr, state_ptr, out_ptr,
    D,
    stride_x_b, stride_state_b, stride_state_d, stride_state_w, stride_out_b,
    WIDTH: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """conv1d causal depthwise width=4 + SiLU + shift del estado, un token.

    ``state[b, d, 0..2]`` son los 3 valores pasados; se desplaza a
    ``(s1, s2, x)`` tras calcular. Los pesos se leen como ``w[d*WIDTH + t]``.
    """
    b = tl.program_id(0)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d < D

    x = tl.load(x_ptr + b * stride_x_b + d, mask=mask, other=0.0).to(tl.float32)
    sb = state_ptr + b * stride_state_b + d * stride_state_d
    s0 = tl.load(sb + 0 * stride_state_w, mask=mask, other=0.0).to(tl.float32)
    s1 = tl.load(sb + 1 * stride_state_w, mask=mask, other=0.0).to(tl.float32)
    s2 = tl.load(sb + 2 * stride_state_w, mask=mask, other=0.0).to(tl.float32)

    wb = w_ptr + d * WIDTH
    y = s0 * tl.load(wb + 0, mask=mask, other=0.0).to(tl.float32)
    y += s1 * tl.load(wb + 1, mask=mask, other=0.0).to(tl.float32)
    y += s2 * tl.load(wb + 2, mask=mask, other=0.0).to(tl.float32)
    y += x * tl.load(wb + 3, mask=mask, other=0.0).to(tl.float32)
    y += tl.load(bias_ptr + d, mask=mask, other=0.0).to(tl.float32)
    y = y * tl.sigmoid(y)

    tl.store(sb + 0 * stride_state_w, s1, mask=mask)
    tl.store(sb + 1 * stride_state_w, s2, mask=mask)
    tl.store(sb + 2 * stride_state_w, x, mask=mask)
    tl.store(out_ptr + b * stride_out_b + d, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _sk08_gated_delta_rule_kernel(
    qkv_ptr, a_ptr, b_ptr, A_log_ptr, dt_bias_ptr, state_ptr, o_ptr,
    stride_qkv_b, stride_a_b, stride_state_b, stride_o_b,
    SCALE: tl.constexpr,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
):
    """Gated delta rule recurrente, una cabeza V por programa."""
    pid = tl.program_id(0)
    i_n = pid // HV
    i_hv = pid % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)

    p_state = state_ptr + i_n * stride_state_b + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(p_state).to(tl.float32)

    p_mixed = qkv_ptr + i_n * stride_qkv_b
    b_q = tl.load(p_mixed + i_h * K + o_k).to(tl.float32)
    b_k = tl.load(p_mixed + H * K + i_h * K + o_k).to(tl.float32)
    b_v = tl.load(p_mixed + 2 * H * K + i_hv * V + o_v).to(tl.float32)

    b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6) * SCALE
    b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)

    a_val = tl.load(a_ptr + i_n * stride_a_b + i_hv).to(tl.float32)
    b_val = tl.load(b_ptr + i_n * stride_a_b + i_hv).to(tl.float32)
    x = a_val + tl.load(dt_bias_ptr + i_hv).to(tl.float32)
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(tl.load(A_log_ptr + i_hv).to(tl.float32)) * softplus_x
    beta_val = tl.sigmoid(b_val)

    b_h *= tl.exp(g_val)
    b_v -= tl.sum(b_h * b_k[None, :], axis=1)
    b_v *= beta_val
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], axis=1)

    tl.store(p_state, b_h.to(state_ptr.dtype.element_ty))
    tl.store(o_ptr + i_n * stride_o_b + i_hv * V + o_v, b_o.to(o_ptr.dtype.element_ty))


def sk08_conv1d_silu(
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    conv_state: torch.Tensor,
) -> torch.Tensor:
    """conv1d causal + SiLU + shift de estado. ``mixed_qkv`` [B, D] -> [B, D]."""
    B, D = mixed_qkv.shape
    out = torch.empty_like(mixed_qkv)
    _lanzar_quant0((B, triton.cdiv(D, 1024)),
            mixed_qkv, conv_weight, conv_bias, conv_state, out, D, mixed_qkv.stride(0), conv_state.stride(0), conv_state.stride(1), conv_state.stride(2), out.stride(0))
    return out


def sk08_gated_delta_rule(
    qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,
    scale: float = 1.0,
    num_k_heads: int = SK08_NUM_K_HEADS,
    num_v_heads: int = SK08_NUM_V_HEADS,
    head_dim: int = SK08_HEAD_DIM,
) -> torch.Tensor:
    """Gated delta rule recurrente. Actualiza ``ssm_state`` in-place, devuelve ``o``."""
    B = qkv.shape[0]
    o = torch.empty((B, num_v_heads, head_dim), dtype=qkv.dtype, device=qkv.device)
    _lanzar_quant1((B * num_v_heads,),
            qkv, a, b, A_log, dt_bias, ssm_state, o, qkv.stride(0), a.stride(0), ssm_state.stride(0), o.stride(0))
    return o


def sk08_ssm_control_bf16_fused(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    scale: float = 1.0,
    num_k_heads: int = SK08_NUM_K_HEADS,
    num_v_heads: int = SK08_NUM_V_HEADS,
    head_dim: int = SK08_HEAD_DIM,
) -> torch.Tensor:
    """Capa completa del control GDN en decode: conv1d+SiLU -> gated delta rule."""
    qkv = sk08_conv1d_silu(mixed_qkv, conv_weight, conv_bias, conv_state)
    return sk08_gated_delta_rule(
        qkv, a, b, A_log, dt_bias, ssm_state, scale, num_k_heads, num_v_heads, head_dim
    )


sk08_ssm_control = sk08_ssm_control_bf16_fused

__all__ = [
    "SK_ID", "SK_NAME", "SK08_CONV_DIM", "SK08_CONV_KERNEL_WIDTH",
    "SK08_NUM_GDN_LAYERS", "SK08_NUM_K_HEADS", "SK08_NUM_V_HEADS", "SK08_HEAD_DIM",
    "SK08_CONV1D_SHAPE", "SK08_A_LOG_SHAPE", "SK08_DT_BIAS_SHAPE",
    "sk08_conv1d_silu", "sk08_gated_delta_rule",
    "sk08_ssm_control_bf16_fused", "sk08_ssm_control",
]


# --- quant: PTX embebido (E1=0 lop3, E2=0 mul) ---

_QPTX0 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk08_conv1d_silu_kernel // -- Begin function _sk08_conv1d_silu_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk08_conv1d_silu_kernel
.visible .entry _sk08_conv1d_silu_kernel(
	.param .u64 .ptr .global .align 1 _sk08_conv1d_silu_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk08_conv1d_silu_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk08_conv1d_silu_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk08_conv1d_silu_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk08_conv1d_silu_kernel_param_4,
	.param .u32 _sk08_conv1d_silu_kernel_param_5,
	.param .u32 _sk08_conv1d_silu_kernel_param_6,
	.param .u32 _sk08_conv1d_silu_kernel_param_7,
	.param .u32 _sk08_conv1d_silu_kernel_param_8,
	.param .u32 _sk08_conv1d_silu_kernel_param_9,
	.param .u64 .ptr .global .align 1 _sk08_conv1d_silu_kernel_param_10,
	.param .u64 .ptr .global .align 1 _sk08_conv1d_silu_kernel_param_11
)
.reqntid 256
{
	.reg .pred 	%p<6>;
	.reg .b16 	%rs<78>;
	.reg .b32 	%r<165>;
	.reg .b64 	%rd<53>;
	.loc	1 78 0                          // sk08_ssm_control.py:78:0
$L__func_begin0:
	.loc	1 78 0                          // sk08_ssm_control.py:78:0

// %bb.0:
	ld.param.b64 	%rd44, [_sk08_conv1d_silu_kernel_param_0];
	ld.param.b64 	%rd45, [_sk08_conv1d_silu_kernel_param_1];
$L__tmp0:
	.loc	1 89 22                         // sk08_ssm_control.py:89:22
	mov.u32 	%r12, %ctaid.x;
	ld.param.b64 	%rd46, [_sk08_conv1d_silu_kernel_param_2];
	.loc	1 90 22                         // sk08_ssm_control.py:90:22
	mov.u32 	%r13, %ctaid.y;
	.loc	1 90 27                         // sk08_ssm_control.py:90:27
	shl.b32 	%r14, %r13, 10;
	ld.param.b64 	%rd47, [_sk08_conv1d_silu_kernel_param_3];
	ld.param.b64 	%rd48, [_sk08_conv1d_silu_kernel_param_4];
	.loc	1 90 50                         // sk08_ssm_control.py:90:50
	mov.u32 	%r15, %tid.x;
	and.b32 	%r16, %r15, 255;
	ld.param.b32 	%r17, [_sk08_conv1d_silu_kernel_param_5];
	shl.b32 	%r18, %r16, 2;
	ld.param.b32 	%r19, [_sk08_conv1d_silu_kernel_param_6];
	.loc	1 90 37                         // sk08_ssm_control.py:90:37
	or.b32 	%r20, %r18, %r14;
	ld.param.b32 	%r21, [_sk08_conv1d_silu_kernel_param_7];
	or.b32 	%r22, %r20, 1;
	ld.param.b32 	%r23, [_sk08_conv1d_silu_kernel_param_8];
	or.b32 	%r24, %r20, 2;
	ld.param.b32 	%r25, [_sk08_conv1d_silu_kernel_param_9];
	or.b32 	%r26, %r20, 3;
	or.b32 	%r27, %r14, %r16;
	or.b32 	%r28, %r27, 256;
	or.b32 	%r29, %r27, 512;
	or.b32 	%r30, %r15, %r14;
	or.b32 	%r31, %r30, 768;
	.loc	1 91 15                         // sk08_ssm_control.py:91:15
	setp.lt.s32 	%p1, %r20, %r17;
	setp.lt.s32 	%p2, %r27, %r17;
	setp.lt.s32 	%p3, %r28, %r17;
	setp.lt.s32 	%p4, %r29, %r17;
	setp.lt.s32 	%p5, %r31, %r17;
	.loc	1 93 28                         // sk08_ssm_control.py:93:28
	mul.lo.s32 	%r32, %r19, %r12;
	.loc	1 93 24                         // sk08_ssm_control.py:93:24
	mad.wide.s32 	%rd49, %r32, 2, %rd44;
	.loc	1 93 41                         // sk08_ssm_control.py:93:41
	mul.wide.u32 	%rd50, %r20, 2;
	add.s64 	%rd1, %rd49, %rd50;
	mov.b32 	%r3, 0;
	.loc	1 93 16                         // sk08_ssm_control.py:93:16
	// begin inline asm
	mov.u32 %r1, %r3;
	mov.u32 %r2, %r3;
	@%p1 ld.global.v2.b32 { %r1, %r2 }, [ %rd1 + 0 ];
	// end inline asm
	.loc	1 94 25                         // sk08_ssm_control.py:94:25
	mul.lo.s32 	%r33, %r21, %r12;
	.loc	1 94 21                         // sk08_ssm_control.py:94:21
	mad.wide.s32 	%rd51, %r33, 2, %rd47;
	.loc	1 94 46                         // sk08_ssm_control.py:94:46
	mul.lo.s32 	%r34, %r23, %r20;
	mul.lo.s32 	%r35, %r23, %r22;
	mul.lo.s32 	%r36, %r23, %r24;
	mul.lo.s32 	%r37, %r23, %r26;
	mul.lo.s32 	%r38, %r23, %r27;
	mul.lo.s32 	%r39, %r23, %r28;
	mul.lo.s32 	%r40, %r23, %r29;
	mul.lo.s32 	%r41, %r23, %r31;
	.loc	1 94 42                         // sk08_ssm_control.py:94:42
	mad.wide.s32 	%rd2, %r34, 2, %rd51;
	mad.wide.s32 	%rd3, %r35, 2, %rd51;
	mad.wide.s32 	%rd4, %r36, 2, %rd51;
	mad.wide.s32 	%rd5, %r37, 2, %rd51;
	mad.wide.s32 	%rd31, %r38, 2, %rd51;
	mad.wide.s32 	%rd32, %r39, 2, %rd51;
	mad.wide.s32 	%rd33, %r40, 2, %rd51;
	mad.wide.s32 	%rd34, %r41, 2, %rd51;
	mov.b16 	%rs2, 0;
	.loc	1 95 17                         // sk08_ssm_control.py:95:17
	// begin inline asm
	mov.u16 %rs1, %rs2;
	@%p1 ld.global.b16 { %rs1 }, [ %rd2 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs3, %rs2;
	@%p1 ld.global.b16 { %rs3 }, [ %rd3 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs4, %rs2;
	@%p1 ld.global.b16 { %rs4 }, [ %rd4 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs5, %rs2;
	@%p1 ld.global.b16 { %rs5 }, [ %rd5 + 0 ];
	// end inline asm
	.loc	1 96 22                         // sk08_ssm_control.py:96:22
	add.s64 	%rd6, %rd2, 2;
	add.s64 	%rd7, %rd3, 2;
	add.s64 	%rd8, %rd4, 2;
	add.s64 	%rd9, %rd5, 2;
	add.s64 	%rd35, %rd31, 2;
	add.s64 	%rd36, %rd32, 2;
	add.s64 	%rd37, %rd33, 2;
	add.s64 	%rd38, %rd34, 2;
	.loc	1 96 17                         // sk08_ssm_control.py:96:17
	// begin inline asm
	mov.u16 %rs22, %rs2;
	@%p1 ld.global.b16 { %rs22 }, [ %rd6 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs23, %rs2;
	@%p1 ld.global.b16 { %rs23 }, [ %rd7 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs24, %rs2;
	@%p1 ld.global.b16 { %rs24 }, [ %rd8 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs25, %rs2;
	@%p1 ld.global.b16 { %rs25 }, [ %rd9 + 0 ];
	// end inline asm
	.loc	1 97 22                         // sk08_ssm_control.py:97:22
	add.s64 	%rd10, %rd2, 4;
	add.s64 	%rd11, %rd3, 4;
	add.s64 	%rd12, %rd4, 4;
	add.s64 	%rd13, %rd5, 4;
	add.s64 	%rd39, %rd31, 4;
	add.s64 	%rd40, %rd32, 4;
	add.s64 	%rd41, %rd33, 4;
	add.s64 	%rd42, %rd34, 4;
	.loc	1 97 17                         // sk08_ssm_control.py:97:17
	// begin inline asm
	mov.u16 %rs30, %rs2;
	@%p1 ld.global.b16 { %rs30 }, [ %rd10 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs31, %rs2;
	@%p1 ld.global.b16 { %rs31 }, [ %rd11 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs32, %rs2;
	@%p1 ld.global.b16 { %rs32 }, [ %rd12 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs33, %rs2;
	@%p1 ld.global.b16 { %rs33 }, [ %rd13 + 0 ];
	// end inline asm
	.loc	1 99 21                         // sk08_ssm_control.py:99:21
	shl.b32 	%r42, %r20, 2;
	shl.b32 	%r43, %r22, 2;
	shl.b32 	%r44, %r24, 2;
	shl.b32 	%r45, %r26, 2;
	.loc	1 99 17                         // sk08_ssm_control.py:99:17
	mad.wide.u32 	%rd14, %r42, 2, %rd45;
	mad.wide.u32 	%rd15, %r43, 2, %rd45;
	mad.wide.u32 	%rd16, %r44, 2, %rd45;
	mad.wide.u32 	%rd17, %r45, 2, %rd45;
	.loc	1 100 21                        // sk08_ssm_control.py:100:21
	// begin inline asm
	mov.u16 %rs6, %rs2;
	@%p1 ld.global.b16 { %rs6 }, [ %rd14 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs7, %rs2;
	@%p1 ld.global.b16 { %rs7 }, [ %rd15 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs8, %rs2;
	@%p1 ld.global.b16 { %rs8 }, [ %rd16 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs9, %rs2;
	@%p1 ld.global.b16 { %rs9 }, [ %rd17 + 0 ];
	// end inline asm
	.loc	1 101 27                        // sk08_ssm_control.py:101:27
	add.s64 	%rd18, %rd14, 2;
	add.s64 	%rd19, %rd15, 2;
	add.s64 	%rd20, %rd16, 2;
	add.s64 	%rd21, %rd17, 2;
	.loc	1 101 22                        // sk08_ssm_control.py:101:22
	// begin inline asm
	mov.u16 %rs10, %rs2;
	@%p1 ld.global.b16 { %rs10 }, [ %rd18 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs11, %rs2;
	@%p1 ld.global.b16 { %rs11 }, [ %rd19 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs12, %rs2;
	@%p1 ld.global.b16 { %rs12 }, [ %rd20 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs13, %rs2;
	@%p1 ld.global.b16 { %rs13 }, [ %rd21 + 0 ];
	// end inline asm
	.loc	1 102 27                        // sk08_ssm_control.py:102:27
	add.s64 	%rd22, %rd14, 4;
	add.s64 	%rd23, %rd15, 4;
	add.s64 	%rd24, %rd16, 4;
	add.s64 	%rd25, %rd17, 4;
	.loc	1 102 22                        // sk08_ssm_control.py:102:22
	// begin inline asm
	mov.u16 %rs14, %rs2;
	@%p1 ld.global.b16 { %rs14 }, [ %rd22 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs15, %rs2;
	@%p1 ld.global.b16 { %rs15 }, [ %rd23 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs16, %rs2;
	@%p1 ld.global.b16 { %rs16 }, [ %rd24 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs17, %rs2;
	@%p1 ld.global.b16 { %rs17 }, [ %rd25 + 0 ];
	// end inline asm
	.loc	1 103 26                        // sk08_ssm_control.py:103:26
	add.s64 	%rd26, %rd14, 6;
	add.s64 	%rd27, %rd15, 6;
	add.s64 	%rd28, %rd16, 6;
	add.s64 	%rd29, %rd17, 6;
	.loc	1 103 21                        // sk08_ssm_control.py:103:21
	// begin inline asm
	mov.u16 %rs18, %rs2;
	@%p1 ld.global.b16 { %rs18 }, [ %rd26 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs19, %rs2;
	@%p1 ld.global.b16 { %rs19 }, [ %rd27 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs20, %rs2;
	@%p1 ld.global.b16 { %rs20 }, [ %rd28 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs21, %rs2;
	@%p1 ld.global.b16 { %rs21 }, [ %rd29 + 0 ];
	// end inline asm
	.loc	1 104 28                        // sk08_ssm_control.py:104:28
	add.s64 	%rd30, %rd46, %rd50;
	.loc	1 104 17                        // sk08_ssm_control.py:104:17
	// begin inline asm
	mov.u32 %r4, %r3;
	mov.u32 %r5, %r3;
	@%p1 ld.global.v2.b32 { %r4, %r5 }, [ %rd30 + 0 ];
	// end inline asm
	.loc	1 107 38                        // sk08_ssm_control.py:107:38
	shl.b32 	%r46, %r15, 2;
	and.b32 	%r47, %r46, 252;
	shr.u32 	%r48, %r15, 5;
	and.b32 	%r49, %r48, 2;
	shl.b32 	%r50, %r15, 1;
	and.b32 	%r51, %r50, 256;
	or.b32 	%r52, %r49, %r51;
	or.b32 	%r53, %r52, %r47;
	mov.b32 	%r54, global_smem;
	add.s32 	%r6, %r54, %r53;
	// begin inline asm
	st.shared.b16 [ %r6 + 0 ], %rs22;
	// end inline asm
	xor.b32 	%r55, %r53, 32;
	add.s32 	%r56, %r54, %r55;
	add.s32 	%r7, %r56, 512;
	// begin inline asm
	st.shared.b16 [ %r7 + 0 ], %rs23;
	// end inline asm
	xor.b32 	%r57, %r53, 64;
	add.s32 	%r58, %r54, %r57;
	add.s32 	%r8, %r58, 1024;
	// begin inline asm
	st.shared.b16 [ %r8 + 0 ], %rs24;
	// end inline asm
	xor.b32 	%r59, %r53, 96;
	add.s32 	%r60, %r54, %r59;
	add.s32 	%r9, %r60, 1536;
	// begin inline asm
	st.shared.b16 [ %r9 + 0 ], %rs25;
	// end inline asm
	bar.sync 	0;
	and.b32 	%r61, %r15, 3;
	shl.b32 	%r62, %r61, 9;
	shl.b32 	%r63, %r61, 5;
	and.b32 	%r64, %r15, 252;
	xor.b32 	%r65, %r63, %r64;
	add.s32 	%r66, %r54, %r62;
	add.s32 	%r67, %r66, %r65;
	ld.shared.v2.b16 	{%rs26, %rs27}, [%r67];
	ld.shared.v2.b16 	{%rs28, %rs29}, [%r67+256];
	// begin inline asm
	@%p2 st.global.b16 [ %rd31 + 0 ], { %rs26 };
	// end inline asm
	// begin inline asm
	@%p3 st.global.b16 [ %rd32 + 0 ], { %rs27 };
	// end inline asm
	// begin inline asm
	@%p4 st.global.b16 [ %rd33 + 0 ], { %rs28 };
	// end inline asm
	// begin inline asm
	@%p5 st.global.b16 [ %rd34 + 0 ], { %rs29 };
	// end inline asm
	.loc	1 108 38                        // sk08_ssm_control.py:108:38
	bar.sync 	0;
	// begin inline asm
	st.shared.b16 [ %r6 + 0 ], %rs30;
	// end inline asm
	// begin inline asm
	st.shared.b16 [ %r7 + 0 ], %rs31;
	// end inline asm
	// begin inline asm
	st.shared.b16 [ %r8 + 0 ], %rs32;
	// end inline asm
	// begin inline asm
	st.shared.b16 [ %r9 + 0 ], %rs33;
	// end inline asm
	bar.sync 	0;
	ld.shared.v2.b16 	{%rs34, %rs35}, [%r67];
	ld.shared.v2.b16 	{%rs36, %rs37}, [%r67+256];
	// begin inline asm
	@%p2 st.global.b16 [ %rd35 + 0 ], { %rs34 };
	// end inline asm
	// begin inline asm
	@%p3 st.global.b16 [ %rd36 + 0 ], { %rs35 };
	// end inline asm
	// begin inline asm
	@%p4 st.global.b16 [ %rd37 + 0 ], { %rs36 };
	// end inline asm
	// begin inline asm
	@%p5 st.global.b16 [ %rd38 + 0 ], { %rs37 };
	// end inline asm
	.loc	1 109 38                        // sk08_ssm_control.py:109:38
	bar.sync 	0;
	mov.b32 	{%rs38, %rs39}, %r1;
	// begin inline asm
	st.shared.b16 [ %r6 + 0 ], %rs38;
	// end inline asm
	// begin inline asm
	st.shared.b16 [ %r7 + 0 ], %rs39;
	// end inline asm
	mov.b32 	{%rs40, %rs41}, %r2;
	// begin inline asm
	st.shared.b16 [ %r8 + 0 ], %rs40;
	// end inline asm
	// begin inline asm
	st.shared.b16 [ %r9 + 0 ], %rs41;
	// end inline asm
	bar.sync 	0;
	ld.shared.v2.b16 	{%rs42, %rs43}, [%r67];
	ld.shared.v2.b16 	{%rs44, %rs45}, [%r67+256];
	// begin inline asm
	@%p2 st.global.b16 [ %rd39 + 0 ], { %rs42 };
	// end inline asm
	// begin inline asm
	@%p3 st.global.b16 [ %rd40 + 0 ], { %rs43 };
	// end inline asm
	// begin inline asm
	@%p4 st.global.b16 [ %rd41 + 0 ], { %rs44 };
	// end inline asm
	// begin inline asm
	@%p5 st.global.b16 [ %rd42 + 0 ], { %rs45 };
	// end inline asm
	.loc	1 110 27                        // sk08_ssm_control.py:110:27
	mul.lo.s32 	%r68, %r25, %r12;
	.loc	1 110 23                        // sk08_ssm_control.py:110:23
	mad.wide.s32 	%rd52, %r68, 2, %rd48;
	.loc	1 110 42                        // sk08_ssm_control.py:110:42
	add.s64 	%rd43, %rd52, %rd50;
	.loc	1 93 69                         // sk08_ssm_control.py:93:69
	cvt.f32.bf16 	%r69, %rs39;
	cvt.f32.bf16 	%r70, %rs38;
	.loc	1 95 17                         // sk08_ssm_control.py:95:17
	mov.b32 	%r71, {%rs1, %rs3};
	.loc	1 95 67                         // sk08_ssm_control.py:95:67
	mov.b32 	{%rs46, %rs47}, %r71;
	cvt.f32.bf16 	%r72, %rs47;
	cvt.f32.bf16 	%r73, %rs46;
	.loc	1 96 17                         // sk08_ssm_control.py:96:17
	mov.b32 	%r74, {%rs22, %rs23};
	.loc	1 96 67                         // sk08_ssm_control.py:96:67
	mov.b32 	{%rs48, %rs49}, %r74;
	cvt.f32.bf16 	%r75, %rs48;
	cvt.f32.bf16 	%r76, %rs49;
	.loc	1 97 17                         // sk08_ssm_control.py:97:17
	mov.b32 	%r77, {%rs30, %rs31};
	.loc	1 97 67                         // sk08_ssm_control.py:97:67
	mov.b32 	{%rs50, %rs51}, %r77;
	cvt.f32.bf16 	%r78, %rs50;
	cvt.f32.bf16 	%r79, %rs51;
	.loc	1 100 21                        // sk08_ssm_control.py:100:21
	mov.b32 	%r80, {%rs6, %rs7};
	.loc	1 100 54                        // sk08_ssm_control.py:100:54
	mov.b32 	{%rs52, %rs53}, %r80;
	cvt.f32.bf16 	%r81, %rs53;
	cvt.f32.bf16 	%r82, %rs52;
	.loc	1 101 22                        // sk08_ssm_control.py:101:22
	mov.b32 	%r83, {%rs10, %rs11};
	.loc	1 101 55                        // sk08_ssm_control.py:101:55
	mov.b32 	{%rs54, %rs55}, %r83;
	cvt.f32.bf16 	%r84, %rs54;
	cvt.f32.bf16 	%r85, %rs55;
	.loc	1 101 14                        // sk08_ssm_control.py:101:14
	mul.f32 	%r86, %r76, %r85;
	mul.f32 	%r87, %r75, %r84;
	.loc	1 101 9                         // sk08_ssm_control.py:101:9
	fma.rn.f32 	%r88, %r73, %r82, %r87;
	fma.rn.f32 	%r89, %r72, %r81, %r86;
	.loc	1 102 22                        // sk08_ssm_control.py:102:22
	mov.b32 	%r90, {%rs14, %rs15};
	.loc	1 102 55                        // sk08_ssm_control.py:102:55
	mov.b32 	{%rs56, %rs57}, %r90;
	cvt.f32.bf16 	%r91, %rs56;
	cvt.f32.bf16 	%r92, %rs57;
	.loc	1 102 9                         // sk08_ssm_control.py:102:9
	fma.rn.f32 	%r93, %r79, %r92, %r89;
	fma.rn.f32 	%r94, %r78, %r91, %r88;
	.loc	1 103 21                        // sk08_ssm_control.py:103:21
	mov.b32 	%r95, {%rs18, %rs19};
	.loc	1 103 54                        // sk08_ssm_control.py:103:54
	mov.b32 	{%rs58, %rs59}, %r95;
	cvt.f32.bf16 	%r96, %rs59;
	cvt.f32.bf16 	%r97, %rs58;
	.loc	1 103 9                         // sk08_ssm_control.py:103:9
	fma.rn.f32 	%r98, %r70, %r97, %r94;
	fma.rn.f32 	%r99, %r69, %r96, %r93;
	.loc	1 104 56                        // sk08_ssm_control.py:104:56
	mov.b32 	{%rs60, %rs61}, %r4;
	cvt.f32.bf16 	%r100, %rs60;
	cvt.f32.bf16 	%r101, %rs61;
	.loc	1 104 9                         // sk08_ssm_control.py:104:9
	add.f32 	%r102, %r99, %r101;
	add.f32 	%r103, %r98, %r100;
	mov.b32 	%r104, 0f00000000;
$L__tmp1:
	.loc	2 50 30                         // standard.py:50:30 @[ sk08_ssm_control.py:105:23 ]
	sub.f32 	%r105, %r104, %r103;
	sub.f32 	%r106, %r104, %r102;
	.loc	2 50 29                         // standard.py:50:29 @[ sk08_ssm_control.py:105:23 ]
	mul.f32 	%r107, %r105, 0f3FB8AA3B;
	ex2.approx.f32 	%r108, %r107;
	mul.f32 	%r109, %r106, 0f3FB8AA3B;
	ex2.approx.f32 	%r110, %r109;
	.loc	2 50 20                         // standard.py:50:20 @[ sk08_ssm_control.py:105:23 ]
	add.f32 	%r111, %r108, 0f3F800000;
	add.f32 	%r112, %r110, 0f3F800000;
	mov.b32 	%r113, 0f3F800000;
	.loc	2 50 16                         // standard.py:50:16 @[ sk08_ssm_control.py:105:23 ]
	div.full.f32 	%r114, %r113, %r111;
	div.full.f32 	%r115, %r113, %r112;
$L__tmp2:
	.loc	1 105 12                        // sk08_ssm_control.py:105:12
	mul.f32 	%r116, %r103, %r114;
	mul.f32 	%r117, %r102, %r115;
	.loc	1 110 50                        // sk08_ssm_control.py:110:50
	cvt.rn.bf16x2.f32 	%r10, %r117, %r116;
	.loc	1 93 69                         // sk08_ssm_control.py:93:69
	cvt.f32.bf16 	%r118, %rs41;
	cvt.f32.bf16 	%r119, %rs40;
	.loc	1 95 17                         // sk08_ssm_control.py:95:17
	mov.b32 	%r120, {%rs4, %rs5};
	.loc	1 95 67                         // sk08_ssm_control.py:95:67
	mov.b32 	{%rs62, %rs63}, %r120;
	cvt.f32.bf16 	%r121, %rs63;
	cvt.f32.bf16 	%r122, %rs62;
	.loc	1 96 17                         // sk08_ssm_control.py:96:17
	mov.b32 	%r123, {%rs24, %rs25};
	.loc	1 96 67                         // sk08_ssm_control.py:96:67
	mov.b32 	{%rs64, %rs65}, %r123;
	cvt.f32.bf16 	%r124, %rs64;
	cvt.f32.bf16 	%r125, %rs65;
	.loc	1 97 17                         // sk08_ssm_control.py:97:17
	mov.b32 	%r126, {%rs32, %rs33};
	.loc	1 97 67                         // sk08_ssm_control.py:97:67
	mov.b32 	{%rs66, %rs67}, %r126;
	cvt.f32.bf16 	%r127, %rs66;
	cvt.f32.bf16 	%r128, %rs67;
	.loc	1 100 21                        // sk08_ssm_control.py:100:21
	mov.b32 	%r129, {%rs8, %rs9};
	.loc	1 100 54                        // sk08_ssm_control.py:100:54
	mov.b32 	{%rs68, %rs69}, %r129;
	cvt.f32.bf16 	%r130, %rs69;
	cvt.f32.bf16 	%r131, %rs68;
	.loc	1 101 22                        // sk08_ssm_control.py:101:22
	mov.b32 	%r132, {%rs12, %rs13};
	.loc	1 101 55                        // sk08_ssm_control.py:101:55
	mov.b32 	{%rs70, %rs71}, %r132;
	cvt.f32.bf16 	%r133, %rs70;
	cvt.f32.bf16 	%r134, %rs71;
	.loc	1 101 14                        // sk08_ssm_control.py:101:14
	mul.f32 	%r135, %r125, %r134;
	mul.f32 	%r136, %r124, %r133;
	.loc	1 101 9                         // sk08_ssm_control.py:101:9
	fma.rn.f32 	%r137, %r122, %r131, %r136;
	fma.rn.f32 	%r138, %r121, %r130, %r135;
	.loc	1 102 22                        // sk08_ssm_control.py:102:22
	mov.b32 	%r139, {%rs16, %rs17};
	.loc	1 102 55                        // sk08_ssm_control.py:102:55
	mov.b32 	{%rs72, %rs73}, %r139;
	cvt.f32.bf16 	%r140, %rs72;
	cvt.f32.bf16 	%r141, %rs73;
	.loc	1 102 9                         // sk08_ssm_control.py:102:9
	fma.rn.f32 	%r142, %r128, %r141, %r138;
	fma.rn.f32 	%r143, %r127, %r140, %r137;
	.loc	1 103 21                        // sk08_ssm_control.py:103:21
	mov.b32 	%r144, {%rs20, %rs21};
	.loc	1 103 54                        // sk08_ssm_control.py:103:54
	mov.b32 	{%rs74, %rs75}, %r144;
	cvt.f32.bf16 	%r145, %rs75;
	cvt.f32.bf16 	%r146, %rs74;
	.loc	1 103 9                         // sk08_ssm_control.py:103:9
	fma.rn.f32 	%r147, %r119, %r146, %r143;
	fma.rn.f32 	%r148, %r118, %r145, %r142;
	.loc	1 104 56                        // sk08_ssm_control.py:104:56
	mov.b32 	{%rs76, %rs77}, %r5;
	cvt.f32.bf16 	%r149, %rs76;
	cvt.f32.bf16 	%r150, %rs77;
	.loc	1 104 9                         // sk08_ssm_control.py:104:9
	add.f32 	%r151, %r148, %r150;
	add.f32 	%r152, %r147, %r149;
$L__tmp3:
	.loc	2 50 30                         // standard.py:50:30 @[ sk08_ssm_control.py:105:23 ]
	sub.f32 	%r153, %r104, %r152;
	sub.f32 	%r154, %r104, %r151;
	.loc	2 50 29                         // standard.py:50:29 @[ sk08_ssm_control.py:105:23 ]
	mul.f32 	%r155, %r153, 0f3FB8AA3B;
	ex2.approx.f32 	%r156, %r155;
	mul.f32 	%r157, %r154, 0f3FB8AA3B;
	ex2.approx.f32 	%r158, %r157;
	.loc	2 50 20                         // standard.py:50:20 @[ sk08_ssm_control.py:105:23 ]
	add.f32 	%r159, %r156, 0f3F800000;
	add.f32 	%r160, %r158, 0f3F800000;
	.loc	2 50 16                         // standard.py:50:16 @[ sk08_ssm_control.py:105:23 ]
	div.full.f32 	%r161, %r113, %r159;
	div.full.f32 	%r162, %r113, %r160;
$L__tmp4:
	.loc	1 105 12                        // sk08_ssm_control.py:105:12
	mul.f32 	%r163, %r152, %r161;
	mul.f32 	%r164, %r151, %r162;
	.loc	1 110 50                        // sk08_ssm_control.py:110:50
	cvt.rn.bf16x2.f32 	%r11, %r164, %r163;
	.loc	1 110 45                        // sk08_ssm_control.py:110:45
	// begin inline asm
	@%p1 st.global.v2.b32 [ %rd43 + 0 ], { %r10, %r11 };
	// end inline asm
	.loc	1 110 4                         // sk08_ssm_control.py:110:4
	ret;
$L__tmp5:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk08_ssm_control.py"
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
.b32 143                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x88 DW_TAG_compile_unit
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
.b8 48
.b8 56
.b8 95
.b8 115
.b8 115
.b8 109
.b8 95
.b8 99
.b8 111
.b8 110
.b8 116
.b8 114
.b8 111
.b8 108
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
.b8 2                                   // Abbrev [2] 0x49:0x1b DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 56
.b8 95
.b8 99
.b8 111
.b8 110
.b8 118
.b8 49
.b8 100
.b8 95
.b8 115
.b8 105
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
.b8 3                                   // Abbrev [3] 0x64:0x2e DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 73                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x79:0x18 DW_TAG_inlined_subroutine
.b32 73                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp4                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 105                                 // DW_AT_call_line
.b8 23                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_QVAR0 = _Nativo(
    "sk08_ssm_control/_sk08_conv1d_silu_kernel",
    _QPTX0, "_sk08_conv1d_silu_kernel",
    warps=8, shared=2048,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10],
    horneado={9: 1, 11: 4, 12: 1024},
    div16=[5, 6, 7, 10],
)


def _q0_impl(grid: list[int], x_ptr: torch.Tensor, w_ptr: torch.Tensor, bias_ptr: torch.Tensor, state_ptr: torch.Tensor, out_ptr: torch.Tensor, D: int, stride_x_b: int, stride_state_b: int, stride_state_d: int, stride_state_w: int, stride_out_b: int, WIDTH: int, BLOCK_D: int) -> None:
    """Cuerpo del custom op: lanza el PTX embebido."""
    _QVAR0(tuple(grid), x_ptr, w_ptr, bias_ptr, state_ptr, out_ptr, D, stride_x_b, stride_state_b, stride_state_d, stride_state_w, stride_out_b, WIDTH, BLOCK_D)


def _q0_fake(grid: list[int], x_ptr: torch.Tensor, w_ptr: torch.Tensor, bias_ptr: torch.Tensor, state_ptr: torch.Tensor, out_ptr: torch.Tensor, D: int, stride_x_b: int, stride_state_b: int, stride_state_d: int, stride_state_w: int, stride_out_b: int, WIDTH: int, BLOCK_D: int) -> None:
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
        op_name="genesis_sk08_ssm_control_q0",
        op_func=_q0_impl,
        mutates_args=['state_ptr', 'out_ptr'],
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
        return torch.ops.vllm.genesis_sk08_ssm_control_q0(g, *args)
    ce = ['WIDTH', 'BLOCK_D']
    nom = ['x_ptr', 'w_ptr', 'bias_ptr', 'state_ptr', 'out_ptr', 'D', 'stride_x_b', 'stride_state_b', 'stride_state_d', 'stride_state_w', 'stride_out_b', 'WIDTH', 'BLOCK_D']
    return _sk08_conv1d_silu_kernel[grid](*args[:len(nom) - len(ce)],
                    **{n: args[nom.index(n)] for n in ce},
                    num_warps=8, num_stages=1)


# --- quant: PTX embebido (E1=0 lop3, E2=0 mul) ---

_QPTX1 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk08_gated_delta_rule_kernel // -- Begin function _sk08_gated_delta_rule_kernel
.extern .shared .align 16 .b8 global_smem[];
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
                                        // @_sk08_gated_delta_rule_kernel
.visible .entry _sk08_gated_delta_rule_kernel(
	.param .u64 .ptr .global .align 1 _sk08_gated_delta_rule_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk08_gated_delta_rule_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk08_gated_delta_rule_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk08_gated_delta_rule_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk08_gated_delta_rule_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk08_gated_delta_rule_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk08_gated_delta_rule_kernel_param_6,
	.param .u32 _sk08_gated_delta_rule_kernel_param_7,
	.param .u32 _sk08_gated_delta_rule_kernel_param_8,
	.param .u32 _sk08_gated_delta_rule_kernel_param_9,
	.param .u32 _sk08_gated_delta_rule_kernel_param_10,
	.param .u64 .ptr .global .align 1 _sk08_gated_delta_rule_kernel_param_11,
	.param .u64 .ptr .global .align 1 _sk08_gated_delta_rule_kernel_param_12
)
.reqntid 128
{
	.reg .pred 	%p<9>;
	.reg .b16 	%rs<41>;
	.reg .b32 	%r<1570>;
	.reg .b64 	%rd<68>;
	.loc	1 114 0                         // sk08_ssm_control.py:114:0
$L__func_begin0:
	.loc	1 114 0                         // sk08_ssm_control.py:114:0

// %bb.0:                               // %__nv_rsqrtf.exit
	ld.param.b32 	%r29, [_sk08_gated_delta_rule_kernel_param_10];
	ld.param.b64 	%rd5, [_sk08_gated_delta_rule_kernel_param_6];
	ld.param.b64 	%rd4, [_sk08_gated_delta_rule_kernel_param_3];
	ld.param.b64 	%rd12, [_sk08_gated_delta_rule_kernel_param_0];
	ld.param.b64 	%rd13, [_sk08_gated_delta_rule_kernel_param_1];
$L__tmp0:
	.loc	1 122 24                        // sk08_ssm_control.py:122:24
	mov.u32 	%r169, %ctaid.x;
	.loc	1 123 17                        // sk08_ssm_control.py:123:17
	mul.hi.u32 	%r170, %r169, 715827883;
	shr.u32 	%r1, %r170, 3;
	ld.param.b64 	%rd14, [_sk08_gated_delta_rule_kernel_param_2];
	.loc	1 124 17                        // sk08_ssm_control.py:124:17
	mul.lo.s32 	%r171, %r1, 48;
	sub.s32 	%r172, %r169, %r171;
	ld.param.b64 	%rd15, [_sk08_gated_delta_rule_kernel_param_4];
	ld.param.b64 	%rd16, [_sk08_gated_delta_rule_kernel_param_5];
	.loc	1 125 19                        // sk08_ssm_control.py:125:19
	cvt.u16.u32 	%rs4, %r172;
	and.b16 	%rs5, %rs4, 255;
	mul.lo.s16 	%rs6, %rs5, 171;
	shr.u16 	%rs7, %rs6, 9;
	ld.param.b32 	%r173, [_sk08_gated_delta_rule_kernel_param_7];
	.loc	1 127 23                        // sk08_ssm_control.py:127:23
	mov.u32 	%r2, %tid.x;
	ld.param.b32 	%r174, [_sk08_gated_delta_rule_kernel_param_8];
	and.b32 	%r175, %r2, 127;
	ld.param.b32 	%r176, [_sk08_gated_delta_rule_kernel_param_9];
	and.b32 	%r177, %r2, 31;
	.loc	1 130 32                        // sk08_ssm_control.py:130:32
	mul.lo.s32 	%r178, %r176, %r1;
	.loc	1 130 26                        // sk08_ssm_control.py:130:26
	mad.wide.s32 	%rd17, %r178, 4, %rd16;
	.loc	1 130 60                        // sk08_ssm_control.py:130:60
	shl.b32 	%r179, %r172, 14;
	.loc	1 130 49                        // sk08_ssm_control.py:130:49
	mad.wide.u32 	%rd18, %r179, 4, %rd17;
	.loc	1 130 68                        // sk08_ssm_control.py:130:68
	and.b32 	%r180, %r2, 96;
	.loc	1 130 79                        // sk08_ssm_control.py:130:79
	shl.b32 	%r181, %r180, 2;
	shl.b32 	%r182, %r2, 2;
	.loc	1 130 64                        // sk08_ssm_control.py:130:64
	mad.wide.u32 	%rd19, %r181, 4, %rd18;
	.loc	1 130 87                        // sk08_ssm_control.py:130:87
	and.b32 	%r3, %r182, 124;
	.loc	1 130 83                        // sk08_ssm_control.py:130:83
	mad.wide.u32 	%rd30, %r3, 4, %rd19;
	add.s64 	%rd31, %rd30, 2048;
	add.s64 	%rd32, %rd30, 4096;
	add.s64 	%rd33, %rd30, 6144;
	add.s64 	%rd34, %rd30, 8192;
	add.s64 	%rd35, %rd30, 10240;
	add.s64 	%rd36, %rd30, 12288;
	add.s64 	%rd37, %rd30, 14336;
	add.s64 	%rd38, %rd30, 16384;
	add.s64 	%rd39, %rd30, 18432;
	add.s64 	%rd40, %rd30, 20480;
	add.s64 	%rd41, %rd30, 22528;
	add.s64 	%rd42, %rd30, 24576;
	add.s64 	%rd43, %rd30, 26624;
	add.s64 	%rd44, %rd30, 28672;
	add.s64 	%rd45, %rd30, 30720;
	add.s64 	%rd46, %rd30, 32768;
	add.s64 	%rd47, %rd30, 34816;
	add.s64 	%rd48, %rd30, 36864;
	add.s64 	%rd49, %rd30, 38912;
	add.s64 	%rd50, %rd30, 40960;
	add.s64 	%rd51, %rd30, 43008;
	add.s64 	%rd52, %rd30, 45056;
	add.s64 	%rd53, %rd30, 47104;
	add.s64 	%rd54, %rd30, 49152;
	add.s64 	%rd55, %rd30, 51200;
	add.s64 	%rd56, %rd30, 53248;
	add.s64 	%rd57, %rd30, 55296;
	add.s64 	%rd58, %rd30, 57344;
	add.s64 	%rd59, %rd30, 59392;
	add.s64 	%rd60, %rd30, 61440;
	add.s64 	%rd61, %rd30, 63488;
	.loc	1 131 18                        // sk08_ssm_control.py:131:18
	// begin inline asm
	mov.u32 %r30, 0x0;
	mov.u32 %r31, 0x0;
	mov.u32 %r32, 0x0;
	mov.u32 %r33, 0x0;
	ld.global.v4.b32 { %r30, %r31, %r32, %r33 }, [ %rd30 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r34, 0x0;
	mov.u32 %r35, 0x0;
	mov.u32 %r36, 0x0;
	mov.u32 %r37, 0x0;
	ld.global.v4.b32 { %r34, %r35, %r36, %r37 }, [ %rd31 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r38, 0x0;
	mov.u32 %r39, 0x0;
	mov.u32 %r40, 0x0;
	mov.u32 %r41, 0x0;
	ld.global.v4.b32 { %r38, %r39, %r40, %r41 }, [ %rd32 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r42, 0x0;
	mov.u32 %r43, 0x0;
	mov.u32 %r44, 0x0;
	mov.u32 %r45, 0x0;
	ld.global.v4.b32 { %r42, %r43, %r44, %r45 }, [ %rd33 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r46, 0x0;
	mov.u32 %r47, 0x0;
	mov.u32 %r48, 0x0;
	mov.u32 %r49, 0x0;
	ld.global.v4.b32 { %r46, %r47, %r48, %r49 }, [ %rd34 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r50, 0x0;
	mov.u32 %r51, 0x0;
	mov.u32 %r52, 0x0;
	mov.u32 %r53, 0x0;
	ld.global.v4.b32 { %r50, %r51, %r52, %r53 }, [ %rd35 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r54, 0x0;
	mov.u32 %r55, 0x0;
	mov.u32 %r56, 0x0;
	mov.u32 %r57, 0x0;
	ld.global.v4.b32 { %r54, %r55, %r56, %r57 }, [ %rd36 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r58, 0x0;
	mov.u32 %r59, 0x0;
	mov.u32 %r60, 0x0;
	mov.u32 %r61, 0x0;
	ld.global.v4.b32 { %r58, %r59, %r60, %r61 }, [ %rd37 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r62, 0x0;
	mov.u32 %r63, 0x0;
	mov.u32 %r64, 0x0;
	mov.u32 %r65, 0x0;
	ld.global.v4.b32 { %r62, %r63, %r64, %r65 }, [ %rd38 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r66, 0x0;
	mov.u32 %r67, 0x0;
	mov.u32 %r68, 0x0;
	mov.u32 %r69, 0x0;
	ld.global.v4.b32 { %r66, %r67, %r68, %r69 }, [ %rd39 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r70, 0x0;
	mov.u32 %r71, 0x0;
	mov.u32 %r72, 0x0;
	mov.u32 %r73, 0x0;
	ld.global.v4.b32 { %r70, %r71, %r72, %r73 }, [ %rd40 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r74, 0x0;
	mov.u32 %r75, 0x0;
	mov.u32 %r76, 0x0;
	mov.u32 %r77, 0x0;
	ld.global.v4.b32 { %r74, %r75, %r76, %r77 }, [ %rd41 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r78, 0x0;
	mov.u32 %r79, 0x0;
	mov.u32 %r80, 0x0;
	mov.u32 %r81, 0x0;
	ld.global.v4.b32 { %r78, %r79, %r80, %r81 }, [ %rd42 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r82, 0x0;
	mov.u32 %r83, 0x0;
	mov.u32 %r84, 0x0;
	mov.u32 %r85, 0x0;
	ld.global.v4.b32 { %r82, %r83, %r84, %r85 }, [ %rd43 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r86, 0x0;
	mov.u32 %r87, 0x0;
	mov.u32 %r88, 0x0;
	mov.u32 %r89, 0x0;
	ld.global.v4.b32 { %r86, %r87, %r88, %r89 }, [ %rd44 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r90, 0x0;
	mov.u32 %r91, 0x0;
	mov.u32 %r92, 0x0;
	mov.u32 %r93, 0x0;
	ld.global.v4.b32 { %r90, %r91, %r92, %r93 }, [ %rd45 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r94, 0x0;
	mov.u32 %r95, 0x0;
	mov.u32 %r96, 0x0;
	mov.u32 %r97, 0x0;
	ld.global.v4.b32 { %r94, %r95, %r96, %r97 }, [ %rd46 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r98, 0x0;
	mov.u32 %r99, 0x0;
	mov.u32 %r100, 0x0;
	mov.u32 %r101, 0x0;
	ld.global.v4.b32 { %r98, %r99, %r100, %r101 }, [ %rd47 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r102, 0x0;
	mov.u32 %r103, 0x0;
	mov.u32 %r104, 0x0;
	mov.u32 %r105, 0x0;
	ld.global.v4.b32 { %r102, %r103, %r104, %r105 }, [ %rd48 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r106, 0x0;
	mov.u32 %r107, 0x0;
	mov.u32 %r108, 0x0;
	mov.u32 %r109, 0x0;
	ld.global.v4.b32 { %r106, %r107, %r108, %r109 }, [ %rd49 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r110, 0x0;
	mov.u32 %r111, 0x0;
	mov.u32 %r112, 0x0;
	mov.u32 %r113, 0x0;
	ld.global.v4.b32 { %r110, %r111, %r112, %r113 }, [ %rd50 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r114, 0x0;
	mov.u32 %r115, 0x0;
	mov.u32 %r116, 0x0;
	mov.u32 %r117, 0x0;
	ld.global.v4.b32 { %r114, %r115, %r116, %r117 }, [ %rd51 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r118, 0x0;
	mov.u32 %r119, 0x0;
	mov.u32 %r120, 0x0;
	mov.u32 %r121, 0x0;
	ld.global.v4.b32 { %r118, %r119, %r120, %r121 }, [ %rd52 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r122, 0x0;
	mov.u32 %r123, 0x0;
	mov.u32 %r124, 0x0;
	mov.u32 %r125, 0x0;
	ld.global.v4.b32 { %r122, %r123, %r124, %r125 }, [ %rd53 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r126, 0x0;
	mov.u32 %r127, 0x0;
	mov.u32 %r128, 0x0;
	mov.u32 %r129, 0x0;
	ld.global.v4.b32 { %r126, %r127, %r128, %r129 }, [ %rd54 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r130, 0x0;
	mov.u32 %r131, 0x0;
	mov.u32 %r132, 0x0;
	mov.u32 %r133, 0x0;
	ld.global.v4.b32 { %r130, %r131, %r132, %r133 }, [ %rd55 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r134, 0x0;
	mov.u32 %r135, 0x0;
	mov.u32 %r136, 0x0;
	mov.u32 %r137, 0x0;
	ld.global.v4.b32 { %r134, %r135, %r136, %r137 }, [ %rd56 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r138, 0x0;
	mov.u32 %r139, 0x0;
	mov.u32 %r140, 0x0;
	mov.u32 %r141, 0x0;
	ld.global.v4.b32 { %r138, %r139, %r140, %r141 }, [ %rd57 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r142, 0x0;
	mov.u32 %r143, 0x0;
	mov.u32 %r144, 0x0;
	mov.u32 %r145, 0x0;
	ld.global.v4.b32 { %r142, %r143, %r144, %r145 }, [ %rd58 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r146, 0x0;
	mov.u32 %r147, 0x0;
	mov.u32 %r148, 0x0;
	mov.u32 %r149, 0x0;
	ld.global.v4.b32 { %r146, %r147, %r148, %r149 }, [ %rd59 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r150, 0x0;
	mov.u32 %r151, 0x0;
	mov.u32 %r152, 0x0;
	mov.u32 %r153, 0x0;
	ld.global.v4.b32 { %r150, %r151, %r152, %r153 }, [ %rd60 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r154, 0x0;
	mov.u32 %r155, 0x0;
	mov.u32 %r156, 0x0;
	mov.u32 %r157, 0x0;
	ld.global.v4.b32 { %r154, %r155, %r156, %r157 }, [ %rd61 + 0 ];
	// end inline asm
	.loc	1 133 30                        // sk08_ssm_control.py:133:30
	mul.lo.s32 	%r183, %r173, %r1;
	.loc	1 133 24                        // sk08_ssm_control.py:133:24
	mad.wide.s32 	%rd20, %r183, 2, %rd12;
	.loc	1 134 28                        // sk08_ssm_control.py:134:28
	cvt.u32.u16 	%r184, %rs7;
	mad.wide.u32 	%rd21, %r184, 256, %rd20;
	.loc	1 134 38                        // sk08_ssm_control.py:134:38
	cvt.u64.u32 	%rd1, %r175;
	mul.wide.u32 	%rd22, %r175, 2;
	add.s64 	%rd6, %rd21, %rd22;
	.loc	1 134 18                        // sk08_ssm_control.py:134:18
	// begin inline asm
	mov.u16 %rs1, 0x0;
	ld.global.b16 { %rs1 }, [ %rd6 + 0 ];
	// end inline asm
	.loc	1 134 46                        // sk08_ssm_control.py:134:46
	cvt.f32.bf16 	%r4, %rs1;
	.loc	1 135 46                        // sk08_ssm_control.py:135:46
	add.s64 	%rd7, %rd6, 4096;
	.loc	1 135 18                        // sk08_ssm_control.py:135:18
	// begin inline asm
	mov.u16 %rs2, 0x0;
	ld.global.b16 { %rs2 }, [ %rd7 + 0 ];
	// end inline asm
	.loc	1 135 54                        // sk08_ssm_control.py:135:54
	cvt.f32.bf16 	%r5, %rs2;
	.loc	1 136 47                        // sk08_ssm_control.py:136:47
	shl.b32 	%r185, %r172, 7;
	.loc	1 136 40                        // sk08_ssm_control.py:136:40
	cvt.u64.u32 	%rd2, %r185;
	mad.wide.u32 	%rd23, %r185, 2, %rd20;
	.loc	1 136 51                        // sk08_ssm_control.py:136:51
	add.s64 	%rd24, %rd23, %rd22;
	add.s64 	%rd8, %rd24, 8192;
	.loc	1 136 18                        // sk08_ssm_control.py:136:18
	// begin inline asm
	mov.u16 %rs3, 0x0;
	ld.global.b16 { %rs3 }, [ %rd8 + 0 ];
	// end inline asm
	.loc	1 150 11                        // sk08_ssm_control.py:150:11
	and.b32 	%r6, %r2, 3;
	shl.b32 	%r7, %r6, 5;
	and.b32 	%r186, %r2, 28;
	shr.u32 	%r187, %r2, 4;
	and.b32 	%r188, %r187, 2;
	shl.b32 	%r189, %r2, 1;
	and.b32 	%r190, %r189, 128;
	mov.b32 	%r191, global_smem;
	add.s32 	%r192, %r191, %r188;
	add.s32 	%r193, %r192, %r190;
	add.s32 	%r194, %r193, %r186;
	add.s32 	%r8, %r194, %r7;
	// begin inline asm
	st.shared.b16 [ %r8 + 0 ], %rs3;
	// end inline asm
	bar.sync 	0;
	add.s32 	%r394, %r191, %r180;
	ld.shared.v4.b32 	{%r9, %r10, %r11, %r12}, [%r394];
	add.s32 	%r399, %r394, 16;
	ld.shared.v4.b32 	{%r13, %r14, %r15, %r16}, [%r394+16];
	add.s32 	%r404, %r394, 128;
	ld.shared.v4.b32 	{%r17, %r18, %r19, %r20}, [%r394+128];
	add.s32 	%r409, %r394, 144;
	ld.shared.v4.b32 	{%r21, %r22, %r23, %r24}, [%r394+144];
	.loc	1 138 38                        // sk08_ssm_control.py:138:38
	mul.f32 	%r195, %r4, %r4;
$L__tmp1:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:138:32 ]
	bar.sync 	0;
	shfl.sync.bfly.b32 	%r196, %r195, 16, 31, -1;
$L__tmp2:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:138:32 ] ]
	fma.rn.f32 	%r197, %r4, %r4, %r196;
$L__tmp3:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:138:32 ]
	shfl.sync.bfly.b32 	%r198, %r197, 8, 31, -1;
$L__tmp4:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:138:32 ] ]
	add.f32 	%r199, %r197, %r198;
$L__tmp5:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:138:32 ]
	shfl.sync.bfly.b32 	%r200, %r199, 4, 31, -1;
$L__tmp6:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:138:32 ] ]
	add.f32 	%r201, %r199, %r200;
$L__tmp7:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:138:32 ]
	shfl.sync.bfly.b32 	%r202, %r201, 2, 31, -1;
$L__tmp8:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:138:32 ] ]
	add.f32 	%r203, %r201, %r202;
$L__tmp9:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:138:32 ]
	shfl.sync.bfly.b32 	%r204, %r203, 1, 31, -1;
$L__tmp10:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:138:32 ] ]
	add.f32 	%r159, %r203, %r204;
$L__tmp11:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:138:32 ]
	setp.eq.b32 	%p1, %r177, 0;
	shr.u32 	%r205, %r2, 3;
	and.b32 	%r206, %r205, 12;
	add.s32 	%r158, %r191, %r206;
	// begin inline asm
	@%p1 st.shared.b32 [ %r158 + 0 ], %r159;
	// end inline asm
	bar.sync 	0;
	setp.lt.u32 	%p2, %r175, 4;
	shl.b32 	%r207, %r175, 2;
	add.s32 	%r161, %r191, %r207;
	// begin inline asm
	@%p2 ld.shared.b32 %r160, [ %r161 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r208, %r160, 2, 31, -1;
$L__tmp12:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:138:32 ] ]
	add.f32 	%r209, %r160, %r208;
$L__tmp13:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:138:32 ]
	shfl.sync.bfly.b32 	%r210, %r209, 1, 31, -1;
$L__tmp14:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:138:32 ] ]
	add.f32 	%r162, %r209, %r210;
$L__tmp15:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:138:32 ]
	setp.eq.b32 	%p4, %r6, 0;
	and.pred 	%p3, %p2, %p4;
	// begin inline asm
	@%p3 st.shared.b32 [ %r161 + 0 ], %r162;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r211, [global_smem];
$L__tmp16:
	.loc	1 138 45                        // sk08_ssm_control.py:138:45
	add.f32 	%r212, %r211, 0f358637BD;
	.loc	1 138 25                        // sk08_ssm_control.py:138:25
	rsqrt.approx.ftz.f32 	%r25, %r212;
	.loc	1 139 38                        // sk08_ssm_control.py:139:38
	mul.f32 	%r213, %r5, %r5;
$L__tmp17:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:139:32 ]
	bar.sync 	0;
	shfl.sync.bfly.b32 	%r214, %r213, 16, 31, -1;
$L__tmp18:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:139:32 ] ]
	fma.rn.f32 	%r215, %r5, %r5, %r214;
$L__tmp19:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:139:32 ]
	shfl.sync.bfly.b32 	%r216, %r215, 8, 31, -1;
$L__tmp20:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:139:32 ] ]
	add.f32 	%r217, %r215, %r216;
$L__tmp21:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:139:32 ]
	shfl.sync.bfly.b32 	%r218, %r217, 4, 31, -1;
$L__tmp22:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:139:32 ] ]
	add.f32 	%r219, %r217, %r218;
$L__tmp23:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:139:32 ]
	shfl.sync.bfly.b32 	%r220, %r219, 2, 31, -1;
$L__tmp24:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:139:32 ] ]
	add.f32 	%r221, %r219, %r220;
$L__tmp25:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:139:32 ]
	shfl.sync.bfly.b32 	%r222, %r221, 1, 31, -1;
$L__tmp26:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:139:32 ] ]
	add.f32 	%r163, %r221, %r222;
$L__tmp27:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:139:32 ]
	// begin inline asm
	@%p1 st.shared.b32 [ %r158 + 0 ], %r163;
	// end inline asm
	bar.sync 	0;
	// begin inline asm
	@%p2 ld.shared.b32 %r164, [ %r161 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r223, %r164, 2, 31, -1;
$L__tmp28:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:139:32 ] ]
	add.f32 	%r224, %r164, %r223;
$L__tmp29:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:139:32 ]
	shfl.sync.bfly.b32 	%r225, %r224, 1, 31, -1;
$L__tmp30:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:139:32 ] ]
	add.f32 	%r165, %r224, %r225;
$L__tmp31:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:139:32 ]
	// begin inline asm
	@%p3 st.shared.b32 [ %r161 + 0 ], %r165;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r226, [global_smem];
$L__tmp32:
	.loc	1 139 45                        // sk08_ssm_control.py:139:45
	add.f32 	%r227, %r226, 0f358637BD;
	.loc	1 139 25                        // sk08_ssm_control.py:139:25
	rsqrt.approx.ftz.f32 	%r26, %r227;
	.loc	1 141 34                        // sk08_ssm_control.py:141:34
	mul.lo.s32 	%r228, %r174, %r1;
	.loc	1 141 28                        // sk08_ssm_control.py:141:28
	mul.wide.s32 	%rd25, %r228, 4;
	add.s64 	%rd26, %rd13, %rd25;
	.loc	1 141 47                        // sk08_ssm_control.py:141:47
	cvt.u64.u32 	%rd3, %r172;
	mul.wide.u32 	%rd27, %r172, 4;
	add.s64 	%rd9, %rd26, %rd27;
	.loc	1 141 20                        // sk08_ssm_control.py:141:20
	// begin inline asm
	mov.u32 %r166, 0x0;
	ld.global.b32 { %r166 }, [ %rd9 + 0 ];
	// end inline asm
	.loc	1 142 28                        // sk08_ssm_control.py:142:28
	add.s64 	%rd28, %rd14, %rd25;
	.loc	1 142 47                        // sk08_ssm_control.py:142:47
	add.s64 	%rd10, %rd28, %rd27;
	.loc	1 142 20                        // sk08_ssm_control.py:142:20
	// begin inline asm
	mov.u32 %r167, 0x0;
	ld.global.b32 { %r167 }, [ %rd10 + 0 ];
	// end inline asm
	.loc	1 143 38                        // sk08_ssm_control.py:143:38
	add.s64 	%rd11, %rd15, %rd27;
	.loc	1 143 24                        // sk08_ssm_control.py:143:24
	// begin inline asm
	mov.u32 %r168, 0x0;
	ld.global.b32 { %r168 }, [ %rd11 + 0 ];
	// end inline asm
	.loc	1 143 16                        // sk08_ssm_control.py:143:16
	add.f32 	%r27, %r166, %r168;
	.loc	1 144 71                        // sk08_ssm_control.py:144:71
	mul.f32 	%r229, %r27, 0f3FB8AA3B;
	ex2.approx.f32 	%r230, %r229;
	.loc	1 144 64                        // sk08_ssm_control.py:144:64
	add.f32 	%r231, %r230, 0f3F800000;
	.loc	1 144 58                        // sk08_ssm_control.py:144:58
	setp.lt.f32 	%p5, %r231, 0f00800000;
	mul.f32 	%r232, %r231, 0f4B000000;
	selp.f32 	%r28, %r232, %r231, %p5;
	selp.f32 	%r233, 0fC1B80000, 0f00000000, %p5;
	add.s32 	%r234, %r28, -1059760811;
	and.b32 	%r235, %r234, -8388608;
	sub.s32 	%r236, %r28, %r235;
	cvt.rn.f32.s32 	%r237, %r235;
	mov.b32 	%r238, 0f34000000;
	fma.rn.ftz.f32 	%r239, %r237, %r238, %r233;
	add.f32 	%r240, %r236, 0fBF800000;
	mov.b32 	%r241, 0f3E1039F6;
	mov.b32 	%r242, 0fBE055027;
	fma.rn.ftz.f32 	%r243, %r242, %r240, %r241;
	mov.b32 	%r244, 0fBDF8CDCC;
	fma.rn.ftz.f32 	%r245, %r243, %r240, %r244;
	mov.b32 	%r246, 0f3E0F2955;
	fma.rn.ftz.f32 	%r247, %r245, %r240, %r246;
	mov.b32 	%r248, 0fBE2AD8B9;
	fma.rn.ftz.f32 	%r249, %r247, %r240, %r248;
	mov.b32 	%r250, 0f3E4CED0B;
	fma.rn.ftz.f32 	%r251, %r249, %r240, %r250;
	mov.b32 	%r252, 0fBE7FFF22;
	fma.rn.ftz.f32 	%r253, %r251, %r240, %r252;
	mov.b32 	%r254, 0f3EAAAA78;
	fma.rn.ftz.f32 	%r255, %r253, %r240, %r254;
	mov.b32 	%r256, 0fBF000000;
	fma.rn.ftz.f32 	%r257, %r255, %r240, %r256;
	mul.f32 	%r258, %r240, %r257;
	fma.rn.ftz.f32 	%r259, %r258, %r240, %r240;
	mov.b32 	%r260, 0f3F317218;
	fma.rn.ftz.f32 	%r1569, %r239, %r260, %r259;
	setp.lt.u32 	%p6, %r28, 2139095040;
	@%p6 bra 	$L__BB0_2;
// %bb.1:                               // %__nv_fmaf_rn.exit.i.i
	.loc	1 0 58                          // sk08_ssm_control.py:0:58
	mov.b32 	%r261, 0f7F800000;
	fma.rn.ftz.f32 	%r1569, %r28, %r261, %r261;
$L__BB0_2:                              // %__nv_logf.exit
	.loc	1 144 31                        // sk08_ssm_control.py:144:31
	setp.le.f32 	%p7, %r27, 0f41A00000;
	.loc	1 139 16                        // sk08_ssm_control.py:139:16
	mul.f32 	%r264, %r26, %r5;
	.loc	1 138 16                        // sk08_ssm_control.py:138:16
	mul.f32 	%r265, %r25, %r4;
	.loc	1 150 11                        // sk08_ssm_control.py:150:11
	mov.b32 	{%rs9, %rs10}, %r24;
	.loc	1 136 59                        // sk08_ssm_control.py:136:59
	cvt.f32.bf16 	%r414, %rs10;
	.loc	1 150 11                        // sk08_ssm_control.py:150:11
	mov.b32 	{%rs11, %rs12}, %r23;
	.loc	1 136 59                        // sk08_ssm_control.py:136:59
	cvt.f32.bf16 	%r415, %rs12;
	.loc	1 150 11                        // sk08_ssm_control.py:150:11
	mov.b32 	{%rs13, %rs14}, %r22;
	.loc	1 136 59                        // sk08_ssm_control.py:136:59
	cvt.f32.bf16 	%r416, %rs14;
	.loc	1 150 11                        // sk08_ssm_control.py:150:11
	mov.b32 	{%rs15, %rs16}, %r21;
	.loc	1 136 59                        // sk08_ssm_control.py:136:59
	cvt.f32.bf16 	%r417, %rs16;
	.loc	1 150 11                        // sk08_ssm_control.py:150:11
	mov.b32 	{%rs17, %rs18}, %r20;
	.loc	1 136 59                        // sk08_ssm_control.py:136:59
	cvt.f32.bf16 	%r418, %rs18;
	.loc	1 150 11                        // sk08_ssm_control.py:150:11
	mov.b32 	{%rs19, %rs20}, %r19;
	.loc	1 136 59                        // sk08_ssm_control.py:136:59
	cvt.f32.bf16 	%r419, %rs20;
	.loc	1 150 11                        // sk08_ssm_control.py:150:11
	mov.b32 	{%rs21, %rs22}, %r18;
	.loc	1 136 59                        // sk08_ssm_control.py:136:59
	cvt.f32.bf16 	%r420, %rs22;
	.loc	1 150 11                        // sk08_ssm_control.py:150:11
	mov.b32 	{%rs23, %rs24}, %r17;
	.loc	1 136 59                        // sk08_ssm_control.py:136:59
	cvt.f32.bf16 	%r421, %rs24;
	cvt.f32.bf16 	%r422, %rs9;
	cvt.f32.bf16 	%r423, %rs11;
	cvt.f32.bf16 	%r424, %rs13;
	cvt.f32.bf16 	%r425, %rs15;
	cvt.f32.bf16 	%r426, %rs17;
	cvt.f32.bf16 	%r427, %rs19;
	cvt.f32.bf16 	%r428, %rs21;
	cvt.f32.bf16 	%r429, %rs23;
	.loc	1 150 11                        // sk08_ssm_control.py:150:11
	mov.b32 	{%rs25, %rs26}, %r16;
	.loc	1 136 59                        // sk08_ssm_control.py:136:59
	cvt.f32.bf16 	%r430, %rs26;
	.loc	1 150 11                        // sk08_ssm_control.py:150:11
	mov.b32 	{%rs27, %rs28}, %r15;
	.loc	1 136 59                        // sk08_ssm_control.py:136:59
	cvt.f32.bf16 	%r431, %rs28;
	.loc	1 150 11                        // sk08_ssm_control.py:150:11
	mov.b32 	{%rs29, %rs30}, %r14;
	.loc	1 136 59                        // sk08_ssm_control.py:136:59
	cvt.f32.bf16 	%r432, %rs30;
	.loc	1 150 11                        // sk08_ssm_control.py:150:11
	mov.b32 	{%rs31, %rs32}, %r13;
	.loc	1 136 59                        // sk08_ssm_control.py:136:59
	cvt.f32.bf16 	%r433, %rs32;
	.loc	1 150 11                        // sk08_ssm_control.py:150:11
	mov.b32 	{%rs33, %rs34}, %r12;
	.loc	1 136 59                        // sk08_ssm_control.py:136:59
	cvt.f32.bf16 	%r434, %rs34;
	.loc	1 150 11                        // sk08_ssm_control.py:150:11
	mov.b32 	{%rs35, %rs36}, %r11;
	.loc	1 136 59                        // sk08_ssm_control.py:136:59
	cvt.f32.bf16 	%r435, %rs36;
	.loc	1 150 11                        // sk08_ssm_control.py:150:11
	mov.b32 	{%rs37, %rs38}, %r10;
	.loc	1 136 59                        // sk08_ssm_control.py:136:59
	cvt.f32.bf16 	%r436, %rs38;
	.loc	1 150 11                        // sk08_ssm_control.py:150:11
	mov.b32 	{%rs39, %rs40}, %r9;
	.loc	1 136 59                        // sk08_ssm_control.py:136:59
	cvt.f32.bf16 	%r437, %rs40;
	cvt.f32.bf16 	%r438, %rs25;
	cvt.f32.bf16 	%r439, %rs27;
	cvt.f32.bf16 	%r440, %rs29;
	cvt.f32.bf16 	%r441, %rs31;
	cvt.f32.bf16 	%r442, %rs33;
	cvt.f32.bf16 	%r443, %rs35;
	cvt.f32.bf16 	%r444, %rs37;
	cvt.f32.bf16 	%r445, %rs39;
	.loc	1 144 58                        // sk08_ssm_control.py:144:58
	setp.eq.f32 	%p8, %r28, 0f00000000;
	selp.f32 	%r446, 0fFF800000, %r1569, %p8;
	.loc	1 144 76                        // sk08_ssm_control.py:144:76
	selp.f32 	%r447, %r446, %r27, %p7;
	.loc	1 145 40                        // sk08_ssm_control.py:145:40
	shl.b64 	%rd63, %rd3, 2;
	add.s64 	%rd29, %rd4, %rd63;
	.loc	1 145 28                        // sk08_ssm_control.py:145:28
	// begin inline asm
	mov.u32 %r262, 0x0;
	ld.global.b32 { %r262 }, [ %rd29 + 0 ];
	// end inline asm
	.loc	1 145 20                        // sk08_ssm_control.py:145:20
	mul.f32 	%r448, %r262, 0f3FB8AA3B;
	ex2.approx.f32 	%r449, %r448;
	mov.b32 	%r450, 0f00000000;
	.loc	1 145 13                        // sk08_ssm_control.py:145:13
	sub.f32 	%r451, %r450, %r449;
	.loc	1 145 64                        // sk08_ssm_control.py:145:64
	mul.f32 	%r452, %r447, %r451;
$L__tmp33:
	.loc	2 50 30                         // standard.py:50:30 @[ sk08_ssm_control.py:146:26 ]
	sub.f32 	%r453, %r450, %r167;
	.loc	2 50 29                         // standard.py:50:29 @[ sk08_ssm_control.py:146:26 ]
	mul.f32 	%r454, %r453, 0f3FB8AA3B;
	ex2.approx.f32 	%r455, %r454;
	.loc	2 50 20                         // standard.py:50:20 @[ sk08_ssm_control.py:146:26 ]
	add.f32 	%r456, %r455, 0f3F800000;
	mov.b32 	%r457, 0f3F800000;
	.loc	2 50 16                         // standard.py:50:16 @[ sk08_ssm_control.py:146:26 ]
	div.full.f32 	%r458, %r457, %r456;
$L__tmp34:
	.loc	1 148 18                        // sk08_ssm_control.py:148:18
	mul.f32 	%r459, %r452, 0f3FB8AA3B;
	ex2.approx.f32 	%r460, %r459;
	.loc	1 148 11                        // sk08_ssm_control.py:148:11
	mul.f32 	%r461, %r460, %r30;
	mul.f32 	%r462, %r460, %r31;
	mul.f32 	%r463, %r460, %r32;
	mul.f32 	%r464, %r460, %r33;
	mul.f32 	%r465, %r460, %r34;
	mul.f32 	%r466, %r460, %r35;
	mul.f32 	%r467, %r460, %r36;
	mul.f32 	%r468, %r460, %r37;
	mul.f32 	%r469, %r460, %r38;
	mul.f32 	%r470, %r460, %r39;
	mul.f32 	%r471, %r460, %r40;
	mul.f32 	%r472, %r460, %r41;
	mul.f32 	%r473, %r460, %r42;
	mul.f32 	%r474, %r460, %r43;
	mul.f32 	%r475, %r460, %r44;
	mul.f32 	%r476, %r460, %r45;
	mul.f32 	%r477, %r460, %r46;
	mul.f32 	%r478, %r460, %r47;
	mul.f32 	%r479, %r460, %r48;
	mul.f32 	%r480, %r460, %r49;
	mul.f32 	%r481, %r460, %r50;
	mul.f32 	%r482, %r460, %r51;
	mul.f32 	%r483, %r460, %r52;
	mul.f32 	%r484, %r460, %r53;
	mul.f32 	%r485, %r460, %r54;
	mul.f32 	%r486, %r460, %r55;
	mul.f32 	%r487, %r460, %r56;
	mul.f32 	%r488, %r460, %r57;
	mul.f32 	%r489, %r460, %r58;
	mul.f32 	%r490, %r460, %r59;
	mul.f32 	%r491, %r460, %r60;
	mul.f32 	%r492, %r460, %r61;
	mul.f32 	%r493, %r460, %r62;
	mul.f32 	%r494, %r460, %r63;
	mul.f32 	%r495, %r460, %r64;
	mul.f32 	%r496, %r460, %r65;
	mul.f32 	%r497, %r460, %r66;
	mul.f32 	%r498, %r460, %r67;
	mul.f32 	%r499, %r460, %r68;
	mul.f32 	%r500, %r460, %r69;
	mul.f32 	%r501, %r460, %r70;
	mul.f32 	%r502, %r460, %r71;
	mul.f32 	%r503, %r460, %r72;
	mul.f32 	%r504, %r460, %r73;
	mul.f32 	%r505, %r460, %r74;
	mul.f32 	%r506, %r460, %r75;
	mul.f32 	%r507, %r460, %r76;
	mul.f32 	%r508, %r460, %r77;
	mul.f32 	%r509, %r460, %r78;
	mul.f32 	%r510, %r460, %r79;
	mul.f32 	%r511, %r460, %r80;
	mul.f32 	%r512, %r460, %r81;
	mul.f32 	%r513, %r460, %r82;
	mul.f32 	%r514, %r460, %r83;
	mul.f32 	%r515, %r460, %r84;
	mul.f32 	%r516, %r460, %r85;
	mul.f32 	%r517, %r460, %r86;
	mul.f32 	%r518, %r460, %r87;
	mul.f32 	%r519, %r460, %r88;
	mul.f32 	%r520, %r460, %r89;
	mul.f32 	%r521, %r460, %r90;
	mul.f32 	%r522, %r460, %r91;
	mul.f32 	%r523, %r460, %r92;
	mul.f32 	%r524, %r460, %r93;
	mul.f32 	%r525, %r460, %r94;
	mul.f32 	%r526, %r460, %r95;
	mul.f32 	%r527, %r460, %r96;
	mul.f32 	%r528, %r460, %r97;
	mul.f32 	%r529, %r460, %r98;
	mul.f32 	%r530, %r460, %r99;
	mul.f32 	%r531, %r460, %r100;
	mul.f32 	%r532, %r460, %r101;
	mul.f32 	%r533, %r460, %r102;
	mul.f32 	%r534, %r460, %r103;
	mul.f32 	%r535, %r460, %r104;
	mul.f32 	%r536, %r460, %r105;
	mul.f32 	%r537, %r460, %r106;
	mul.f32 	%r538, %r460, %r107;
	mul.f32 	%r539, %r460, %r108;
	mul.f32 	%r540, %r460, %r109;
	mul.f32 	%r541, %r460, %r110;
	mul.f32 	%r542, %r460, %r111;
	mul.f32 	%r543, %r460, %r112;
	mul.f32 	%r544, %r460, %r113;
	mul.f32 	%r545, %r460, %r114;
	mul.f32 	%r546, %r460, %r115;
	mul.f32 	%r547, %r460, %r116;
	mul.f32 	%r548, %r460, %r117;
	mul.f32 	%r549, %r460, %r118;
	mul.f32 	%r550, %r460, %r119;
	mul.f32 	%r551, %r460, %r120;
	mul.f32 	%r552, %r460, %r121;
	mul.f32 	%r553, %r460, %r122;
	mul.f32 	%r554, %r460, %r123;
	mul.f32 	%r555, %r460, %r124;
	mul.f32 	%r556, %r460, %r125;
	mul.f32 	%r557, %r460, %r126;
	mul.f32 	%r558, %r460, %r127;
	mul.f32 	%r559, %r460, %r128;
	mul.f32 	%r560, %r460, %r129;
	mul.f32 	%r561, %r460, %r130;
	mul.f32 	%r562, %r460, %r131;
	mul.f32 	%r563, %r460, %r132;
	mul.f32 	%r564, %r460, %r133;
	mul.f32 	%r565, %r460, %r134;
	mul.f32 	%r566, %r460, %r135;
	mul.f32 	%r567, %r460, %r136;
	mul.f32 	%r568, %r460, %r137;
	mul.f32 	%r569, %r460, %r138;
	mul.f32 	%r570, %r460, %r139;
	mul.f32 	%r571, %r460, %r140;
	mul.f32 	%r572, %r460, %r141;
	mul.f32 	%r573, %r460, %r142;
	mul.f32 	%r574, %r460, %r143;
	mul.f32 	%r575, %r460, %r144;
	mul.f32 	%r576, %r460, %r145;
	mul.f32 	%r577, %r460, %r146;
	mul.f32 	%r578, %r460, %r147;
	mul.f32 	%r579, %r460, %r148;
	mul.f32 	%r580, %r460, %r149;
	mul.f32 	%r581, %r460, %r150;
	mul.f32 	%r582, %r460, %r151;
	mul.f32 	%r583, %r460, %r152;
	mul.f32 	%r584, %r460, %r153;
	mul.f32 	%r585, %r460, %r154;
	mul.f32 	%r586, %r460, %r155;
	mul.f32 	%r587, %r460, %r156;
	mul.f32 	%r588, %r460, %r157;
	.loc	1 149 24                        // sk08_ssm_control.py:149:24
	bar.sync 	0;
	shl.b32 	%r589, %r6, 7;
	and.b32 	%r590, %r2, 124;
	xor.b32 	%r591, %r7, %r590;
	add.s32 	%r592, %r191, %r589;
	add.s32 	%r263, %r592, %r591;
	// begin inline asm
	st.shared.b32 [ %r263 + 0 ], %r264;
	// end inline asm
	bar.sync 	0;
	add.s32 	%r593, %r191, %r3;
	ld.shared.b32 	%r594, [%r593];
	xor.b32 	%r595, %r3, 32;
	add.s32 	%r596, %r191, %r595;
	ld.shared.b32 	%r597, [%r596+128];
	xor.b32 	%r598, %r3, 64;
	add.s32 	%r599, %r191, %r598;
	ld.shared.b32 	%r600, [%r599+256];
	xor.b32 	%r601, %r3, 96;
	add.s32 	%r602, %r191, %r601;
	ld.shared.b32 	%r603, [%r602+384];
	mul.f32 	%r604, %r462, %r597;
	mul.f32 	%r605, %r466, %r597;
	mul.f32 	%r606, %r470, %r597;
	mul.f32 	%r607, %r474, %r597;
	mul.f32 	%r608, %r478, %r597;
	mul.f32 	%r609, %r482, %r597;
	mul.f32 	%r610, %r486, %r597;
	mul.f32 	%r611, %r490, %r597;
	mul.f32 	%r612, %r494, %r597;
	mul.f32 	%r613, %r498, %r597;
	mul.f32 	%r614, %r502, %r597;
	mul.f32 	%r615, %r506, %r597;
	mul.f32 	%r616, %r510, %r597;
	mul.f32 	%r617, %r514, %r597;
	mul.f32 	%r618, %r518, %r597;
	mul.f32 	%r619, %r522, %r597;
	mul.f32 	%r620, %r526, %r597;
	mul.f32 	%r621, %r530, %r597;
	mul.f32 	%r622, %r534, %r597;
	mul.f32 	%r623, %r538, %r597;
	mul.f32 	%r624, %r542, %r597;
	mul.f32 	%r625, %r546, %r597;
	mul.f32 	%r626, %r550, %r597;
	mul.f32 	%r627, %r554, %r597;
	mul.f32 	%r628, %r558, %r597;
	mul.f32 	%r629, %r562, %r597;
	mul.f32 	%r630, %r566, %r597;
	mul.f32 	%r631, %r570, %r597;
	mul.f32 	%r632, %r574, %r597;
	mul.f32 	%r633, %r578, %r597;
	mul.f32 	%r634, %r582, %r597;
	mul.f32 	%r635, %r586, %r597;
$L__tmp35:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	fma.rn.f32 	%r636, %r461, %r594, %r604;
	fma.rn.f32 	%r637, %r463, %r600, %r636;
	fma.rn.f32 	%r638, %r464, %r603, %r637;
	fma.rn.f32 	%r639, %r465, %r594, %r605;
	fma.rn.f32 	%r640, %r467, %r600, %r639;
	fma.rn.f32 	%r641, %r468, %r603, %r640;
	fma.rn.f32 	%r642, %r469, %r594, %r606;
	fma.rn.f32 	%r643, %r471, %r600, %r642;
	fma.rn.f32 	%r644, %r472, %r603, %r643;
	fma.rn.f32 	%r645, %r473, %r594, %r607;
	fma.rn.f32 	%r646, %r475, %r600, %r645;
	fma.rn.f32 	%r647, %r476, %r603, %r646;
	fma.rn.f32 	%r648, %r477, %r594, %r608;
	fma.rn.f32 	%r649, %r479, %r600, %r648;
	fma.rn.f32 	%r650, %r480, %r603, %r649;
	fma.rn.f32 	%r651, %r481, %r594, %r609;
	fma.rn.f32 	%r652, %r483, %r600, %r651;
	fma.rn.f32 	%r653, %r484, %r603, %r652;
	fma.rn.f32 	%r654, %r485, %r594, %r610;
	fma.rn.f32 	%r655, %r487, %r600, %r654;
	fma.rn.f32 	%r656, %r488, %r603, %r655;
	fma.rn.f32 	%r657, %r489, %r594, %r611;
	fma.rn.f32 	%r658, %r491, %r600, %r657;
	fma.rn.f32 	%r659, %r492, %r603, %r658;
	fma.rn.f32 	%r660, %r493, %r594, %r612;
	fma.rn.f32 	%r661, %r495, %r600, %r660;
	fma.rn.f32 	%r662, %r496, %r603, %r661;
	fma.rn.f32 	%r663, %r497, %r594, %r613;
	fma.rn.f32 	%r664, %r499, %r600, %r663;
	fma.rn.f32 	%r665, %r500, %r603, %r664;
	fma.rn.f32 	%r666, %r501, %r594, %r614;
	fma.rn.f32 	%r667, %r503, %r600, %r666;
	fma.rn.f32 	%r668, %r504, %r603, %r667;
	fma.rn.f32 	%r669, %r505, %r594, %r615;
	fma.rn.f32 	%r670, %r507, %r600, %r669;
	fma.rn.f32 	%r671, %r508, %r603, %r670;
	fma.rn.f32 	%r672, %r509, %r594, %r616;
	fma.rn.f32 	%r673, %r511, %r600, %r672;
	fma.rn.f32 	%r674, %r512, %r603, %r673;
	fma.rn.f32 	%r675, %r513, %r594, %r617;
	fma.rn.f32 	%r676, %r515, %r600, %r675;
	fma.rn.f32 	%r677, %r516, %r603, %r676;
	fma.rn.f32 	%r678, %r517, %r594, %r618;
	fma.rn.f32 	%r679, %r519, %r600, %r678;
	fma.rn.f32 	%r680, %r520, %r603, %r679;
	fma.rn.f32 	%r681, %r521, %r594, %r619;
	fma.rn.f32 	%r682, %r523, %r600, %r681;
	fma.rn.f32 	%r683, %r524, %r603, %r682;
	fma.rn.f32 	%r684, %r525, %r594, %r620;
	fma.rn.f32 	%r685, %r527, %r600, %r684;
	fma.rn.f32 	%r686, %r528, %r603, %r685;
	fma.rn.f32 	%r687, %r529, %r594, %r621;
	fma.rn.f32 	%r688, %r531, %r600, %r687;
	fma.rn.f32 	%r689, %r532, %r603, %r688;
	fma.rn.f32 	%r690, %r533, %r594, %r622;
	fma.rn.f32 	%r691, %r535, %r600, %r690;
	fma.rn.f32 	%r692, %r536, %r603, %r691;
	fma.rn.f32 	%r693, %r537, %r594, %r623;
	fma.rn.f32 	%r694, %r539, %r600, %r693;
	fma.rn.f32 	%r695, %r540, %r603, %r694;
	fma.rn.f32 	%r696, %r541, %r594, %r624;
	fma.rn.f32 	%r697, %r543, %r600, %r696;
	fma.rn.f32 	%r698, %r544, %r603, %r697;
	fma.rn.f32 	%r699, %r545, %r594, %r625;
	fma.rn.f32 	%r700, %r547, %r600, %r699;
	fma.rn.f32 	%r701, %r548, %r603, %r700;
	fma.rn.f32 	%r702, %r549, %r594, %r626;
	fma.rn.f32 	%r703, %r551, %r600, %r702;
	fma.rn.f32 	%r704, %r552, %r603, %r703;
	fma.rn.f32 	%r705, %r553, %r594, %r627;
	fma.rn.f32 	%r706, %r555, %r600, %r705;
	fma.rn.f32 	%r707, %r556, %r603, %r706;
	fma.rn.f32 	%r708, %r557, %r594, %r628;
	fma.rn.f32 	%r709, %r559, %r600, %r708;
	fma.rn.f32 	%r710, %r560, %r603, %r709;
	fma.rn.f32 	%r711, %r561, %r594, %r629;
	fma.rn.f32 	%r712, %r563, %r600, %r711;
	fma.rn.f32 	%r713, %r564, %r603, %r712;
	fma.rn.f32 	%r714, %r565, %r594, %r630;
	fma.rn.f32 	%r715, %r567, %r600, %r714;
	fma.rn.f32 	%r716, %r568, %r603, %r715;
	fma.rn.f32 	%r717, %r569, %r594, %r631;
	fma.rn.f32 	%r718, %r571, %r600, %r717;
	fma.rn.f32 	%r719, %r572, %r603, %r718;
	fma.rn.f32 	%r720, %r573, %r594, %r632;
	fma.rn.f32 	%r721, %r575, %r600, %r720;
	fma.rn.f32 	%r722, %r576, %r603, %r721;
	fma.rn.f32 	%r723, %r577, %r594, %r633;
	fma.rn.f32 	%r724, %r579, %r600, %r723;
	fma.rn.f32 	%r725, %r580, %r603, %r724;
	fma.rn.f32 	%r726, %r581, %r594, %r634;
	fma.rn.f32 	%r727, %r583, %r600, %r726;
	fma.rn.f32 	%r728, %r584, %r603, %r727;
	fma.rn.f32 	%r729, %r585, %r594, %r635;
	fma.rn.f32 	%r730, %r587, %r600, %r729;
	fma.rn.f32 	%r731, %r588, %r603, %r730;
$L__tmp36:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r732, %r638, 16, 31, -1;
$L__tmp37:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r733, %r638, %r732;
$L__tmp38:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r734, %r733, 8, 31, -1;
$L__tmp39:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r735, %r733, %r734;
$L__tmp40:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r736, %r735, 4, 31, -1;
$L__tmp41:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r737, %r735, %r736;
$L__tmp42:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r738, %r737, 2, 31, -1;
$L__tmp43:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r739, %r737, %r738;
$L__tmp44:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r740, %r739, 1, 31, -1;
$L__tmp45:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r741, %r739, %r740;
$L__tmp46:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r742, %r641, 16, 31, -1;
$L__tmp47:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r743, %r641, %r742;
$L__tmp48:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r744, %r743, 8, 31, -1;
$L__tmp49:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r745, %r743, %r744;
$L__tmp50:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r746, %r745, 4, 31, -1;
$L__tmp51:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r747, %r745, %r746;
$L__tmp52:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r748, %r747, 2, 31, -1;
$L__tmp53:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r749, %r747, %r748;
$L__tmp54:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r750, %r749, 1, 31, -1;
$L__tmp55:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r751, %r749, %r750;
$L__tmp56:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r752, %r644, 16, 31, -1;
$L__tmp57:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r753, %r644, %r752;
$L__tmp58:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r754, %r753, 8, 31, -1;
$L__tmp59:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r755, %r753, %r754;
$L__tmp60:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r756, %r755, 4, 31, -1;
$L__tmp61:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r757, %r755, %r756;
$L__tmp62:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r758, %r757, 2, 31, -1;
$L__tmp63:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r759, %r757, %r758;
$L__tmp64:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r760, %r759, 1, 31, -1;
$L__tmp65:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r761, %r759, %r760;
$L__tmp66:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r762, %r647, 16, 31, -1;
$L__tmp67:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r763, %r647, %r762;
$L__tmp68:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r764, %r763, 8, 31, -1;
$L__tmp69:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r765, %r763, %r764;
$L__tmp70:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r766, %r765, 4, 31, -1;
$L__tmp71:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r767, %r765, %r766;
$L__tmp72:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r768, %r767, 2, 31, -1;
$L__tmp73:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r769, %r767, %r768;
$L__tmp74:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r770, %r769, 1, 31, -1;
$L__tmp75:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r771, %r769, %r770;
$L__tmp76:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r772, %r650, 16, 31, -1;
$L__tmp77:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r773, %r650, %r772;
$L__tmp78:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r774, %r773, 8, 31, -1;
$L__tmp79:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r775, %r773, %r774;
$L__tmp80:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r776, %r775, 4, 31, -1;
$L__tmp81:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r777, %r775, %r776;
$L__tmp82:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r778, %r777, 2, 31, -1;
$L__tmp83:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r779, %r777, %r778;
$L__tmp84:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r780, %r779, 1, 31, -1;
$L__tmp85:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r781, %r779, %r780;
$L__tmp86:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r782, %r653, 16, 31, -1;
$L__tmp87:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r783, %r653, %r782;
$L__tmp88:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r784, %r783, 8, 31, -1;
$L__tmp89:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r785, %r783, %r784;
$L__tmp90:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r786, %r785, 4, 31, -1;
$L__tmp91:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r787, %r785, %r786;
$L__tmp92:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r788, %r787, 2, 31, -1;
$L__tmp93:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r789, %r787, %r788;
$L__tmp94:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r790, %r789, 1, 31, -1;
$L__tmp95:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r791, %r789, %r790;
$L__tmp96:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r792, %r656, 16, 31, -1;
$L__tmp97:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r793, %r656, %r792;
$L__tmp98:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r794, %r793, 8, 31, -1;
$L__tmp99:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r795, %r793, %r794;
$L__tmp100:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r796, %r795, 4, 31, -1;
$L__tmp101:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r797, %r795, %r796;
$L__tmp102:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r798, %r797, 2, 31, -1;
$L__tmp103:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r799, %r797, %r798;
$L__tmp104:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r800, %r799, 1, 31, -1;
$L__tmp105:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r801, %r799, %r800;
$L__tmp106:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r802, %r659, 16, 31, -1;
$L__tmp107:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r803, %r659, %r802;
$L__tmp108:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r804, %r803, 8, 31, -1;
$L__tmp109:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r805, %r803, %r804;
$L__tmp110:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r806, %r805, 4, 31, -1;
$L__tmp111:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r807, %r805, %r806;
$L__tmp112:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r808, %r807, 2, 31, -1;
$L__tmp113:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r809, %r807, %r808;
$L__tmp114:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r810, %r809, 1, 31, -1;
$L__tmp115:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r811, %r809, %r810;
$L__tmp116:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r812, %r662, 16, 31, -1;
$L__tmp117:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r813, %r662, %r812;
$L__tmp118:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r814, %r813, 8, 31, -1;
$L__tmp119:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r815, %r813, %r814;
$L__tmp120:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r816, %r815, 4, 31, -1;
$L__tmp121:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r817, %r815, %r816;
$L__tmp122:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r818, %r817, 2, 31, -1;
$L__tmp123:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r819, %r817, %r818;
$L__tmp124:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r820, %r819, 1, 31, -1;
$L__tmp125:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r821, %r819, %r820;
$L__tmp126:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r822, %r665, 16, 31, -1;
$L__tmp127:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r823, %r665, %r822;
$L__tmp128:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r824, %r823, 8, 31, -1;
$L__tmp129:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r825, %r823, %r824;
$L__tmp130:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r826, %r825, 4, 31, -1;
$L__tmp131:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r827, %r825, %r826;
$L__tmp132:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r828, %r827, 2, 31, -1;
$L__tmp133:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r829, %r827, %r828;
$L__tmp134:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r830, %r829, 1, 31, -1;
$L__tmp135:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r831, %r829, %r830;
$L__tmp136:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r832, %r668, 16, 31, -1;
$L__tmp137:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r833, %r668, %r832;
$L__tmp138:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r834, %r833, 8, 31, -1;
$L__tmp139:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r835, %r833, %r834;
$L__tmp140:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r836, %r835, 4, 31, -1;
$L__tmp141:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r837, %r835, %r836;
$L__tmp142:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r838, %r837, 2, 31, -1;
$L__tmp143:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r839, %r837, %r838;
$L__tmp144:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r840, %r839, 1, 31, -1;
$L__tmp145:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r841, %r839, %r840;
$L__tmp146:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r842, %r671, 16, 31, -1;
$L__tmp147:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r843, %r671, %r842;
$L__tmp148:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r844, %r843, 8, 31, -1;
$L__tmp149:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r845, %r843, %r844;
$L__tmp150:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r846, %r845, 4, 31, -1;
$L__tmp151:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r847, %r845, %r846;
$L__tmp152:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r848, %r847, 2, 31, -1;
$L__tmp153:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r849, %r847, %r848;
$L__tmp154:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r850, %r849, 1, 31, -1;
$L__tmp155:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r851, %r849, %r850;
$L__tmp156:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r852, %r674, 16, 31, -1;
$L__tmp157:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r853, %r674, %r852;
$L__tmp158:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r854, %r853, 8, 31, -1;
$L__tmp159:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r855, %r853, %r854;
$L__tmp160:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r856, %r855, 4, 31, -1;
$L__tmp161:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r857, %r855, %r856;
$L__tmp162:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r858, %r857, 2, 31, -1;
$L__tmp163:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r859, %r857, %r858;
$L__tmp164:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r860, %r859, 1, 31, -1;
$L__tmp165:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r861, %r859, %r860;
$L__tmp166:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r862, %r677, 16, 31, -1;
$L__tmp167:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r863, %r677, %r862;
$L__tmp168:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r864, %r863, 8, 31, -1;
$L__tmp169:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r865, %r863, %r864;
$L__tmp170:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r866, %r865, 4, 31, -1;
$L__tmp171:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r867, %r865, %r866;
$L__tmp172:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r868, %r867, 2, 31, -1;
$L__tmp173:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r869, %r867, %r868;
$L__tmp174:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r870, %r869, 1, 31, -1;
$L__tmp175:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r871, %r869, %r870;
$L__tmp176:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r872, %r680, 16, 31, -1;
$L__tmp177:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r873, %r680, %r872;
$L__tmp178:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r874, %r873, 8, 31, -1;
$L__tmp179:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r875, %r873, %r874;
$L__tmp180:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r876, %r875, 4, 31, -1;
$L__tmp181:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r877, %r875, %r876;
$L__tmp182:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r878, %r877, 2, 31, -1;
$L__tmp183:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r879, %r877, %r878;
$L__tmp184:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r880, %r879, 1, 31, -1;
$L__tmp185:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r881, %r879, %r880;
$L__tmp186:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r882, %r683, 16, 31, -1;
$L__tmp187:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r883, %r683, %r882;
$L__tmp188:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r884, %r883, 8, 31, -1;
$L__tmp189:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r885, %r883, %r884;
$L__tmp190:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r886, %r885, 4, 31, -1;
$L__tmp191:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r887, %r885, %r886;
$L__tmp192:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r888, %r887, 2, 31, -1;
$L__tmp193:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r889, %r887, %r888;
$L__tmp194:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r890, %r889, 1, 31, -1;
$L__tmp195:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r891, %r889, %r890;
$L__tmp196:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r892, %r686, 16, 31, -1;
$L__tmp197:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r893, %r686, %r892;
$L__tmp198:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r894, %r893, 8, 31, -1;
$L__tmp199:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r895, %r893, %r894;
$L__tmp200:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r896, %r895, 4, 31, -1;
$L__tmp201:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r897, %r895, %r896;
$L__tmp202:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r898, %r897, 2, 31, -1;
$L__tmp203:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r899, %r897, %r898;
$L__tmp204:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r900, %r899, 1, 31, -1;
$L__tmp205:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r901, %r899, %r900;
$L__tmp206:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r902, %r689, 16, 31, -1;
$L__tmp207:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r903, %r689, %r902;
$L__tmp208:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r904, %r903, 8, 31, -1;
$L__tmp209:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r905, %r903, %r904;
$L__tmp210:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r906, %r905, 4, 31, -1;
$L__tmp211:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r907, %r905, %r906;
$L__tmp212:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r908, %r907, 2, 31, -1;
$L__tmp213:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r909, %r907, %r908;
$L__tmp214:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r910, %r909, 1, 31, -1;
$L__tmp215:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r911, %r909, %r910;
$L__tmp216:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r912, %r692, 16, 31, -1;
$L__tmp217:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r913, %r692, %r912;
$L__tmp218:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r914, %r913, 8, 31, -1;
$L__tmp219:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r915, %r913, %r914;
$L__tmp220:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r916, %r915, 4, 31, -1;
$L__tmp221:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r917, %r915, %r916;
$L__tmp222:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r918, %r917, 2, 31, -1;
$L__tmp223:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r919, %r917, %r918;
$L__tmp224:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r920, %r919, 1, 31, -1;
$L__tmp225:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r921, %r919, %r920;
$L__tmp226:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r922, %r695, 16, 31, -1;
$L__tmp227:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r923, %r695, %r922;
$L__tmp228:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r924, %r923, 8, 31, -1;
$L__tmp229:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r925, %r923, %r924;
$L__tmp230:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r926, %r925, 4, 31, -1;
$L__tmp231:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r927, %r925, %r926;
$L__tmp232:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r928, %r927, 2, 31, -1;
$L__tmp233:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r929, %r927, %r928;
$L__tmp234:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r930, %r929, 1, 31, -1;
$L__tmp235:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r931, %r929, %r930;
$L__tmp236:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r932, %r698, 16, 31, -1;
$L__tmp237:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r933, %r698, %r932;
$L__tmp238:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r934, %r933, 8, 31, -1;
$L__tmp239:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r935, %r933, %r934;
$L__tmp240:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r936, %r935, 4, 31, -1;
$L__tmp241:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r937, %r935, %r936;
$L__tmp242:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r938, %r937, 2, 31, -1;
$L__tmp243:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r939, %r937, %r938;
$L__tmp244:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r940, %r939, 1, 31, -1;
$L__tmp245:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r941, %r939, %r940;
$L__tmp246:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r942, %r701, 16, 31, -1;
$L__tmp247:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r943, %r701, %r942;
$L__tmp248:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r944, %r943, 8, 31, -1;
$L__tmp249:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r945, %r943, %r944;
$L__tmp250:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r946, %r945, 4, 31, -1;
$L__tmp251:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r947, %r945, %r946;
$L__tmp252:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r948, %r947, 2, 31, -1;
$L__tmp253:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r949, %r947, %r948;
$L__tmp254:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r950, %r949, 1, 31, -1;
$L__tmp255:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r951, %r949, %r950;
$L__tmp256:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r952, %r704, 16, 31, -1;
$L__tmp257:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r953, %r704, %r952;
$L__tmp258:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r954, %r953, 8, 31, -1;
$L__tmp259:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r955, %r953, %r954;
$L__tmp260:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r956, %r955, 4, 31, -1;
$L__tmp261:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r957, %r955, %r956;
$L__tmp262:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r958, %r957, 2, 31, -1;
$L__tmp263:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r959, %r957, %r958;
$L__tmp264:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r960, %r959, 1, 31, -1;
$L__tmp265:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r961, %r959, %r960;
$L__tmp266:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r962, %r707, 16, 31, -1;
$L__tmp267:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r963, %r707, %r962;
$L__tmp268:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r964, %r963, 8, 31, -1;
$L__tmp269:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r965, %r963, %r964;
$L__tmp270:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r966, %r965, 4, 31, -1;
$L__tmp271:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r967, %r965, %r966;
$L__tmp272:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r968, %r967, 2, 31, -1;
$L__tmp273:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r969, %r967, %r968;
$L__tmp274:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r970, %r969, 1, 31, -1;
$L__tmp275:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r971, %r969, %r970;
$L__tmp276:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r972, %r710, 16, 31, -1;
$L__tmp277:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r973, %r710, %r972;
$L__tmp278:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r974, %r973, 8, 31, -1;
$L__tmp279:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r975, %r973, %r974;
$L__tmp280:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r976, %r975, 4, 31, -1;
$L__tmp281:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r977, %r975, %r976;
$L__tmp282:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r978, %r977, 2, 31, -1;
$L__tmp283:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r979, %r977, %r978;
$L__tmp284:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r980, %r979, 1, 31, -1;
$L__tmp285:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r981, %r979, %r980;
$L__tmp286:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r982, %r713, 16, 31, -1;
$L__tmp287:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r983, %r713, %r982;
$L__tmp288:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r984, %r983, 8, 31, -1;
$L__tmp289:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r985, %r983, %r984;
$L__tmp290:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r986, %r985, 4, 31, -1;
$L__tmp291:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r987, %r985, %r986;
$L__tmp292:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r988, %r987, 2, 31, -1;
$L__tmp293:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r989, %r987, %r988;
$L__tmp294:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r990, %r989, 1, 31, -1;
$L__tmp295:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r991, %r989, %r990;
$L__tmp296:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r992, %r716, 16, 31, -1;
$L__tmp297:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r993, %r716, %r992;
$L__tmp298:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r994, %r993, 8, 31, -1;
$L__tmp299:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r995, %r993, %r994;
$L__tmp300:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r996, %r995, 4, 31, -1;
$L__tmp301:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r997, %r995, %r996;
$L__tmp302:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r998, %r997, 2, 31, -1;
$L__tmp303:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r999, %r997, %r998;
$L__tmp304:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r1000, %r999, 1, 31, -1;
$L__tmp305:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r1001, %r999, %r1000;
$L__tmp306:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r1002, %r719, 16, 31, -1;
$L__tmp307:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r1003, %r719, %r1002;
$L__tmp308:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r1004, %r1003, 8, 31, -1;
$L__tmp309:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r1005, %r1003, %r1004;
$L__tmp310:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r1006, %r1005, 4, 31, -1;
$L__tmp311:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r1007, %r1005, %r1006;
$L__tmp312:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r1008, %r1007, 2, 31, -1;
$L__tmp313:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r1009, %r1007, %r1008;
$L__tmp314:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r1010, %r1009, 1, 31, -1;
$L__tmp315:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r1011, %r1009, %r1010;
$L__tmp316:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r1012, %r722, 16, 31, -1;
$L__tmp317:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r1013, %r722, %r1012;
$L__tmp318:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r1014, %r1013, 8, 31, -1;
$L__tmp319:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r1015, %r1013, %r1014;
$L__tmp320:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r1016, %r1015, 4, 31, -1;
$L__tmp321:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r1017, %r1015, %r1016;
$L__tmp322:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r1018, %r1017, 2, 31, -1;
$L__tmp323:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r1019, %r1017, %r1018;
$L__tmp324:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r1020, %r1019, 1, 31, -1;
$L__tmp325:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r1021, %r1019, %r1020;
$L__tmp326:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r1022, %r725, 16, 31, -1;
$L__tmp327:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r1023, %r725, %r1022;
$L__tmp328:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r1024, %r1023, 8, 31, -1;
$L__tmp329:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r1025, %r1023, %r1024;
$L__tmp330:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r1026, %r1025, 4, 31, -1;
$L__tmp331:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r1027, %r1025, %r1026;
$L__tmp332:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r1028, %r1027, 2, 31, -1;
$L__tmp333:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r1029, %r1027, %r1028;
$L__tmp334:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r1030, %r1029, 1, 31, -1;
$L__tmp335:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r1031, %r1029, %r1030;
$L__tmp336:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r1032, %r728, 16, 31, -1;
$L__tmp337:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r1033, %r728, %r1032;
$L__tmp338:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r1034, %r1033, 8, 31, -1;
$L__tmp339:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r1035, %r1033, %r1034;
$L__tmp340:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r1036, %r1035, 4, 31, -1;
$L__tmp341:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r1037, %r1035, %r1036;
$L__tmp342:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r1038, %r1037, 2, 31, -1;
$L__tmp343:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r1039, %r1037, %r1038;
$L__tmp344:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r1040, %r1039, 1, 31, -1;
$L__tmp345:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r1041, %r1039, %r1040;
$L__tmp346:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r1042, %r731, 16, 31, -1;
$L__tmp347:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r1043, %r731, %r1042;
$L__tmp348:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r1044, %r1043, 8, 31, -1;
$L__tmp349:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r1045, %r1043, %r1044;
$L__tmp350:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r1046, %r1045, 4, 31, -1;
$L__tmp351:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r1047, %r1045, %r1046;
$L__tmp352:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r1048, %r1047, 2, 31, -1;
$L__tmp353:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r1049, %r1047, %r1048;
$L__tmp354:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:149:18 ]
	shfl.sync.bfly.b32 	%r1050, %r1049, 1, 31, -1;
$L__tmp355:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:149:18 ] ]
	add.f32 	%r1051, %r1049, %r1050;
$L__tmp356:
	.loc	1 149 11                        // sk08_ssm_control.py:149:11
	sub.f32 	%r1052, %r445, %r741;
	sub.f32 	%r1053, %r444, %r751;
	sub.f32 	%r1054, %r443, %r761;
	sub.f32 	%r1055, %r442, %r771;
	sub.f32 	%r1056, %r441, %r781;
	sub.f32 	%r1057, %r440, %r791;
	sub.f32 	%r1058, %r439, %r801;
	sub.f32 	%r1059, %r438, %r811;
	sub.f32 	%r1060, %r437, %r821;
	sub.f32 	%r1061, %r436, %r831;
	sub.f32 	%r1062, %r435, %r841;
	sub.f32 	%r1063, %r434, %r851;
	sub.f32 	%r1064, %r433, %r861;
	sub.f32 	%r1065, %r432, %r871;
	sub.f32 	%r1066, %r431, %r881;
	sub.f32 	%r1067, %r430, %r891;
	sub.f32 	%r1068, %r429, %r901;
	sub.f32 	%r1069, %r428, %r911;
	sub.f32 	%r1070, %r427, %r921;
	sub.f32 	%r1071, %r426, %r931;
	sub.f32 	%r1072, %r425, %r941;
	sub.f32 	%r1073, %r424, %r951;
	sub.f32 	%r1074, %r423, %r961;
	sub.f32 	%r1075, %r422, %r971;
	sub.f32 	%r1076, %r421, %r981;
	sub.f32 	%r1077, %r420, %r991;
	sub.f32 	%r1078, %r419, %r1001;
	sub.f32 	%r1079, %r418, %r1011;
	sub.f32 	%r1080, %r417, %r1021;
	sub.f32 	%r1081, %r416, %r1031;
	sub.f32 	%r1082, %r415, %r1041;
	sub.f32 	%r1083, %r414, %r1051;
	.loc	1 150 11                        // sk08_ssm_control.py:150:11
	mul.f32 	%r1084, %r458, %r1052;
	mul.f32 	%r1085, %r458, %r1053;
	mul.f32 	%r1086, %r458, %r1054;
	mul.f32 	%r1087, %r458, %r1055;
	mul.f32 	%r1088, %r458, %r1056;
	mul.f32 	%r1089, %r458, %r1057;
	mul.f32 	%r1090, %r458, %r1058;
	mul.f32 	%r1091, %r458, %r1059;
	mul.f32 	%r1092, %r458, %r1060;
	mul.f32 	%r1093, %r458, %r1061;
	mul.f32 	%r1094, %r458, %r1062;
	mul.f32 	%r1095, %r458, %r1063;
	mul.f32 	%r1096, %r458, %r1064;
	mul.f32 	%r1097, %r458, %r1065;
	mul.f32 	%r1098, %r458, %r1066;
	mul.f32 	%r1099, %r458, %r1067;
	mul.f32 	%r1100, %r458, %r1068;
	mul.f32 	%r1101, %r458, %r1069;
	mul.f32 	%r1102, %r458, %r1070;
	mul.f32 	%r1103, %r458, %r1071;
	mul.f32 	%r1104, %r458, %r1072;
	mul.f32 	%r1105, %r458, %r1073;
	mul.f32 	%r1106, %r458, %r1074;
	mul.f32 	%r1107, %r458, %r1075;
	mul.f32 	%r1108, %r458, %r1076;
	mul.f32 	%r1109, %r458, %r1077;
	mul.f32 	%r1110, %r458, %r1078;
	mul.f32 	%r1111, %r458, %r1079;
	mul.f32 	%r1112, %r458, %r1080;
	mul.f32 	%r1113, %r458, %r1081;
	mul.f32 	%r1114, %r458, %r1082;
	mul.f32 	%r1115, %r458, %r1083;
	.loc	1 151 11                        // sk08_ssm_control.py:151:11
	fma.rn.f32 	%r266, %r594, %r1084, %r461;
	fma.rn.f32 	%r267, %r597, %r1084, %r462;
	fma.rn.f32 	%r268, %r600, %r1084, %r463;
	fma.rn.f32 	%r269, %r603, %r1084, %r464;
	fma.rn.f32 	%r270, %r594, %r1085, %r465;
	fma.rn.f32 	%r271, %r597, %r1085, %r466;
	fma.rn.f32 	%r272, %r600, %r1085, %r467;
	fma.rn.f32 	%r273, %r603, %r1085, %r468;
	fma.rn.f32 	%r274, %r594, %r1086, %r469;
	fma.rn.f32 	%r275, %r597, %r1086, %r470;
	fma.rn.f32 	%r276, %r600, %r1086, %r471;
	fma.rn.f32 	%r277, %r603, %r1086, %r472;
	fma.rn.f32 	%r278, %r594, %r1087, %r473;
	fma.rn.f32 	%r279, %r597, %r1087, %r474;
	fma.rn.f32 	%r280, %r600, %r1087, %r475;
	fma.rn.f32 	%r281, %r603, %r1087, %r476;
	fma.rn.f32 	%r282, %r594, %r1088, %r477;
	fma.rn.f32 	%r283, %r597, %r1088, %r478;
	fma.rn.f32 	%r284, %r600, %r1088, %r479;
	fma.rn.f32 	%r285, %r603, %r1088, %r480;
	fma.rn.f32 	%r286, %r594, %r1089, %r481;
	fma.rn.f32 	%r287, %r597, %r1089, %r482;
	fma.rn.f32 	%r288, %r600, %r1089, %r483;
	fma.rn.f32 	%r289, %r603, %r1089, %r484;
	fma.rn.f32 	%r290, %r594, %r1090, %r485;
	fma.rn.f32 	%r291, %r597, %r1090, %r486;
	fma.rn.f32 	%r292, %r600, %r1090, %r487;
	fma.rn.f32 	%r293, %r603, %r1090, %r488;
	fma.rn.f32 	%r294, %r594, %r1091, %r489;
	fma.rn.f32 	%r295, %r597, %r1091, %r490;
	fma.rn.f32 	%r296, %r600, %r1091, %r491;
	fma.rn.f32 	%r297, %r603, %r1091, %r492;
	fma.rn.f32 	%r298, %r594, %r1092, %r493;
	fma.rn.f32 	%r299, %r597, %r1092, %r494;
	fma.rn.f32 	%r300, %r600, %r1092, %r495;
	fma.rn.f32 	%r301, %r603, %r1092, %r496;
	fma.rn.f32 	%r302, %r594, %r1093, %r497;
	fma.rn.f32 	%r303, %r597, %r1093, %r498;
	fma.rn.f32 	%r304, %r600, %r1093, %r499;
	fma.rn.f32 	%r305, %r603, %r1093, %r500;
	fma.rn.f32 	%r306, %r594, %r1094, %r501;
	fma.rn.f32 	%r307, %r597, %r1094, %r502;
	fma.rn.f32 	%r308, %r600, %r1094, %r503;
	fma.rn.f32 	%r309, %r603, %r1094, %r504;
	fma.rn.f32 	%r310, %r594, %r1095, %r505;
	fma.rn.f32 	%r311, %r597, %r1095, %r506;
	fma.rn.f32 	%r312, %r600, %r1095, %r507;
	fma.rn.f32 	%r313, %r603, %r1095, %r508;
	fma.rn.f32 	%r314, %r594, %r1096, %r509;
	fma.rn.f32 	%r315, %r597, %r1096, %r510;
	fma.rn.f32 	%r316, %r600, %r1096, %r511;
	fma.rn.f32 	%r317, %r603, %r1096, %r512;
	fma.rn.f32 	%r318, %r594, %r1097, %r513;
	fma.rn.f32 	%r319, %r597, %r1097, %r514;
	fma.rn.f32 	%r320, %r600, %r1097, %r515;
	fma.rn.f32 	%r321, %r603, %r1097, %r516;
	fma.rn.f32 	%r322, %r594, %r1098, %r517;
	fma.rn.f32 	%r323, %r597, %r1098, %r518;
	fma.rn.f32 	%r324, %r600, %r1098, %r519;
	fma.rn.f32 	%r325, %r603, %r1098, %r520;
	fma.rn.f32 	%r326, %r594, %r1099, %r521;
	fma.rn.f32 	%r327, %r597, %r1099, %r522;
	fma.rn.f32 	%r328, %r600, %r1099, %r523;
	fma.rn.f32 	%r329, %r603, %r1099, %r524;
	fma.rn.f32 	%r330, %r594, %r1100, %r525;
	fma.rn.f32 	%r331, %r597, %r1100, %r526;
	fma.rn.f32 	%r332, %r600, %r1100, %r527;
	fma.rn.f32 	%r333, %r603, %r1100, %r528;
	fma.rn.f32 	%r334, %r594, %r1101, %r529;
	fma.rn.f32 	%r335, %r597, %r1101, %r530;
	fma.rn.f32 	%r336, %r600, %r1101, %r531;
	fma.rn.f32 	%r337, %r603, %r1101, %r532;
	fma.rn.f32 	%r338, %r594, %r1102, %r533;
	fma.rn.f32 	%r339, %r597, %r1102, %r534;
	fma.rn.f32 	%r340, %r600, %r1102, %r535;
	fma.rn.f32 	%r341, %r603, %r1102, %r536;
	fma.rn.f32 	%r342, %r594, %r1103, %r537;
	fma.rn.f32 	%r343, %r597, %r1103, %r538;
	fma.rn.f32 	%r344, %r600, %r1103, %r539;
	fma.rn.f32 	%r345, %r603, %r1103, %r540;
	fma.rn.f32 	%r346, %r594, %r1104, %r541;
	fma.rn.f32 	%r347, %r597, %r1104, %r542;
	fma.rn.f32 	%r348, %r600, %r1104, %r543;
	fma.rn.f32 	%r349, %r603, %r1104, %r544;
	fma.rn.f32 	%r350, %r594, %r1105, %r545;
	fma.rn.f32 	%r351, %r597, %r1105, %r546;
	fma.rn.f32 	%r352, %r600, %r1105, %r547;
	fma.rn.f32 	%r353, %r603, %r1105, %r548;
	fma.rn.f32 	%r354, %r594, %r1106, %r549;
	fma.rn.f32 	%r355, %r597, %r1106, %r550;
	fma.rn.f32 	%r356, %r600, %r1106, %r551;
	fma.rn.f32 	%r357, %r603, %r1106, %r552;
	fma.rn.f32 	%r358, %r594, %r1107, %r553;
	fma.rn.f32 	%r359, %r597, %r1107, %r554;
	fma.rn.f32 	%r360, %r600, %r1107, %r555;
	fma.rn.f32 	%r361, %r603, %r1107, %r556;
	fma.rn.f32 	%r362, %r594, %r1108, %r557;
	fma.rn.f32 	%r363, %r597, %r1108, %r558;
	fma.rn.f32 	%r364, %r600, %r1108, %r559;
	fma.rn.f32 	%r365, %r603, %r1108, %r560;
	fma.rn.f32 	%r366, %r594, %r1109, %r561;
	fma.rn.f32 	%r367, %r597, %r1109, %r562;
	fma.rn.f32 	%r368, %r600, %r1109, %r563;
	fma.rn.f32 	%r369, %r603, %r1109, %r564;
	fma.rn.f32 	%r370, %r594, %r1110, %r565;
	fma.rn.f32 	%r371, %r597, %r1110, %r566;
	fma.rn.f32 	%r372, %r600, %r1110, %r567;
	fma.rn.f32 	%r373, %r603, %r1110, %r568;
	fma.rn.f32 	%r374, %r594, %r1111, %r569;
	fma.rn.f32 	%r375, %r597, %r1111, %r570;
	fma.rn.f32 	%r376, %r600, %r1111, %r571;
	fma.rn.f32 	%r377, %r603, %r1111, %r572;
	fma.rn.f32 	%r378, %r594, %r1112, %r573;
	fma.rn.f32 	%r379, %r597, %r1112, %r574;
	fma.rn.f32 	%r380, %r600, %r1112, %r575;
	fma.rn.f32 	%r381, %r603, %r1112, %r576;
	fma.rn.f32 	%r382, %r594, %r1113, %r577;
	fma.rn.f32 	%r383, %r597, %r1113, %r578;
	fma.rn.f32 	%r384, %r600, %r1113, %r579;
	fma.rn.f32 	%r385, %r603, %r1113, %r580;
	fma.rn.f32 	%r386, %r594, %r1114, %r581;
	fma.rn.f32 	%r387, %r597, %r1114, %r582;
	fma.rn.f32 	%r388, %r600, %r1114, %r583;
	fma.rn.f32 	%r389, %r603, %r1114, %r584;
	fma.rn.f32 	%r390, %r594, %r1115, %r585;
	fma.rn.f32 	%r391, %r597, %r1115, %r586;
	fma.rn.f32 	%r392, %r600, %r1115, %r587;
	fma.rn.f32 	%r393, %r603, %r1115, %r588;
	.loc	1 152 23                        // sk08_ssm_control.py:152:23
	bar.sync 	0;
	// begin inline asm
	st.shared.b32 [ %r263 + 0 ], %r265;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r1116, [%r593];
	ld.shared.b32 	%r1117, [%r596+128];
	ld.shared.b32 	%r1118, [%r599+256];
	ld.shared.b32 	%r1119, [%r602+384];
	mul.f32 	%r1120, %r267, %r1117;
	mul.f32 	%r1121, %r271, %r1117;
	mul.f32 	%r1122, %r275, %r1117;
	mul.f32 	%r1123, %r279, %r1117;
	mul.f32 	%r1124, %r283, %r1117;
	mul.f32 	%r1125, %r287, %r1117;
	mul.f32 	%r1126, %r291, %r1117;
	mul.f32 	%r1127, %r295, %r1117;
	mul.f32 	%r1128, %r299, %r1117;
	mul.f32 	%r1129, %r303, %r1117;
	mul.f32 	%r1130, %r307, %r1117;
	mul.f32 	%r1131, %r311, %r1117;
	mul.f32 	%r1132, %r315, %r1117;
	mul.f32 	%r1133, %r319, %r1117;
	mul.f32 	%r1134, %r323, %r1117;
	mul.f32 	%r1135, %r327, %r1117;
	mul.f32 	%r1136, %r331, %r1117;
	mul.f32 	%r1137, %r335, %r1117;
	mul.f32 	%r1138, %r339, %r1117;
	mul.f32 	%r1139, %r343, %r1117;
	mul.f32 	%r1140, %r347, %r1117;
	mul.f32 	%r1141, %r351, %r1117;
	mul.f32 	%r1142, %r355, %r1117;
	mul.f32 	%r1143, %r359, %r1117;
	mul.f32 	%r1144, %r363, %r1117;
	mul.f32 	%r1145, %r367, %r1117;
	mul.f32 	%r1146, %r371, %r1117;
	mul.f32 	%r1147, %r375, %r1117;
	mul.f32 	%r1148, %r379, %r1117;
	mul.f32 	%r1149, %r383, %r1117;
	mul.f32 	%r1150, %r387, %r1117;
	mul.f32 	%r1151, %r1117, %r391;
$L__tmp357:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	fma.rn.f32 	%r1152, %r266, %r1116, %r1120;
	fma.rn.f32 	%r1153, %r268, %r1118, %r1152;
	fma.rn.f32 	%r1154, %r269, %r1119, %r1153;
	fma.rn.f32 	%r1155, %r270, %r1116, %r1121;
	fma.rn.f32 	%r1156, %r272, %r1118, %r1155;
	fma.rn.f32 	%r1157, %r273, %r1119, %r1156;
	fma.rn.f32 	%r1158, %r274, %r1116, %r1122;
	fma.rn.f32 	%r1159, %r276, %r1118, %r1158;
	fma.rn.f32 	%r1160, %r277, %r1119, %r1159;
	fma.rn.f32 	%r1161, %r278, %r1116, %r1123;
	fma.rn.f32 	%r1162, %r280, %r1118, %r1161;
	fma.rn.f32 	%r1163, %r281, %r1119, %r1162;
	fma.rn.f32 	%r1164, %r282, %r1116, %r1124;
	fma.rn.f32 	%r1165, %r284, %r1118, %r1164;
	fma.rn.f32 	%r1166, %r285, %r1119, %r1165;
	fma.rn.f32 	%r1167, %r286, %r1116, %r1125;
	fma.rn.f32 	%r1168, %r288, %r1118, %r1167;
	fma.rn.f32 	%r1169, %r289, %r1119, %r1168;
	fma.rn.f32 	%r1170, %r290, %r1116, %r1126;
	fma.rn.f32 	%r1171, %r292, %r1118, %r1170;
	fma.rn.f32 	%r1172, %r293, %r1119, %r1171;
	fma.rn.f32 	%r1173, %r294, %r1116, %r1127;
	fma.rn.f32 	%r1174, %r296, %r1118, %r1173;
	fma.rn.f32 	%r1175, %r297, %r1119, %r1174;
	fma.rn.f32 	%r1176, %r298, %r1116, %r1128;
	fma.rn.f32 	%r1177, %r300, %r1118, %r1176;
	fma.rn.f32 	%r1178, %r301, %r1119, %r1177;
	fma.rn.f32 	%r1179, %r302, %r1116, %r1129;
	fma.rn.f32 	%r1180, %r304, %r1118, %r1179;
	fma.rn.f32 	%r1181, %r305, %r1119, %r1180;
	fma.rn.f32 	%r1182, %r306, %r1116, %r1130;
	fma.rn.f32 	%r1183, %r308, %r1118, %r1182;
	fma.rn.f32 	%r1184, %r309, %r1119, %r1183;
	fma.rn.f32 	%r1185, %r310, %r1116, %r1131;
	fma.rn.f32 	%r1186, %r312, %r1118, %r1185;
	fma.rn.f32 	%r1187, %r313, %r1119, %r1186;
	fma.rn.f32 	%r1188, %r314, %r1116, %r1132;
	fma.rn.f32 	%r1189, %r316, %r1118, %r1188;
	fma.rn.f32 	%r1190, %r317, %r1119, %r1189;
	fma.rn.f32 	%r1191, %r318, %r1116, %r1133;
	fma.rn.f32 	%r1192, %r320, %r1118, %r1191;
	fma.rn.f32 	%r1193, %r321, %r1119, %r1192;
	fma.rn.f32 	%r1194, %r322, %r1116, %r1134;
	fma.rn.f32 	%r1195, %r324, %r1118, %r1194;
	fma.rn.f32 	%r1196, %r325, %r1119, %r1195;
	fma.rn.f32 	%r1197, %r326, %r1116, %r1135;
	fma.rn.f32 	%r1198, %r328, %r1118, %r1197;
	fma.rn.f32 	%r1199, %r329, %r1119, %r1198;
	fma.rn.f32 	%r1200, %r330, %r1116, %r1136;
	fma.rn.f32 	%r1201, %r332, %r1118, %r1200;
	fma.rn.f32 	%r1202, %r333, %r1119, %r1201;
	fma.rn.f32 	%r1203, %r334, %r1116, %r1137;
	fma.rn.f32 	%r1204, %r336, %r1118, %r1203;
	fma.rn.f32 	%r1205, %r337, %r1119, %r1204;
	fma.rn.f32 	%r1206, %r338, %r1116, %r1138;
	fma.rn.f32 	%r1207, %r340, %r1118, %r1206;
	fma.rn.f32 	%r1208, %r341, %r1119, %r1207;
	fma.rn.f32 	%r1209, %r342, %r1116, %r1139;
	fma.rn.f32 	%r1210, %r344, %r1118, %r1209;
	fma.rn.f32 	%r1211, %r345, %r1119, %r1210;
	fma.rn.f32 	%r1212, %r346, %r1116, %r1140;
	fma.rn.f32 	%r1213, %r348, %r1118, %r1212;
	fma.rn.f32 	%r1214, %r349, %r1119, %r1213;
	fma.rn.f32 	%r1215, %r350, %r1116, %r1141;
	fma.rn.f32 	%r1216, %r352, %r1118, %r1215;
	fma.rn.f32 	%r1217, %r353, %r1119, %r1216;
	fma.rn.f32 	%r1218, %r354, %r1116, %r1142;
	fma.rn.f32 	%r1219, %r356, %r1118, %r1218;
	fma.rn.f32 	%r1220, %r357, %r1119, %r1219;
	fma.rn.f32 	%r1221, %r358, %r1116, %r1143;
	fma.rn.f32 	%r1222, %r360, %r1118, %r1221;
	fma.rn.f32 	%r1223, %r361, %r1119, %r1222;
	fma.rn.f32 	%r1224, %r362, %r1116, %r1144;
	fma.rn.f32 	%r1225, %r364, %r1118, %r1224;
	fma.rn.f32 	%r1226, %r365, %r1119, %r1225;
	fma.rn.f32 	%r1227, %r366, %r1116, %r1145;
	fma.rn.f32 	%r1228, %r368, %r1118, %r1227;
	fma.rn.f32 	%r1229, %r369, %r1119, %r1228;
	fma.rn.f32 	%r1230, %r370, %r1116, %r1146;
	fma.rn.f32 	%r1231, %r372, %r1118, %r1230;
	fma.rn.f32 	%r1232, %r373, %r1119, %r1231;
	fma.rn.f32 	%r1233, %r374, %r1116, %r1147;
	fma.rn.f32 	%r1234, %r376, %r1118, %r1233;
	fma.rn.f32 	%r1235, %r377, %r1119, %r1234;
	fma.rn.f32 	%r1236, %r378, %r1116, %r1148;
	fma.rn.f32 	%r1237, %r380, %r1118, %r1236;
	fma.rn.f32 	%r1238, %r381, %r1119, %r1237;
	fma.rn.f32 	%r1239, %r382, %r1116, %r1149;
	fma.rn.f32 	%r1240, %r384, %r1118, %r1239;
	fma.rn.f32 	%r1241, %r385, %r1119, %r1240;
	fma.rn.f32 	%r1242, %r386, %r1116, %r1150;
	fma.rn.f32 	%r1243, %r388, %r1118, %r1242;
	fma.rn.f32 	%r1244, %r389, %r1119, %r1243;
	fma.rn.f32 	%r1245, %r1116, %r390, %r1151;
	fma.rn.f32 	%r1246, %r392, %r1118, %r1245;
	fma.rn.f32 	%r1247, %r393, %r1119, %r1246;
$L__tmp358:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1248, %r1154, 16, 31, -1;
$L__tmp359:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1249, %r1154, %r1248;
$L__tmp360:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1250, %r1249, 8, 31, -1;
$L__tmp361:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1251, %r1249, %r1250;
$L__tmp362:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1252, %r1251, 4, 31, -1;
$L__tmp363:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1253, %r1251, %r1252;
$L__tmp364:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1254, %r1253, 2, 31, -1;
$L__tmp365:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1255, %r1253, %r1254;
$L__tmp366:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1256, %r1255, 1, 31, -1;
$L__tmp367:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1257, %r1255, %r1256;
$L__tmp368:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1258, %r1157, 16, 31, -1;
$L__tmp369:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1259, %r1157, %r1258;
$L__tmp370:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1260, %r1259, 8, 31, -1;
$L__tmp371:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1261, %r1259, %r1260;
$L__tmp372:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1262, %r1261, 4, 31, -1;
$L__tmp373:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1263, %r1261, %r1262;
$L__tmp374:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1264, %r1263, 2, 31, -1;
$L__tmp375:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1265, %r1263, %r1264;
$L__tmp376:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1266, %r1265, 1, 31, -1;
$L__tmp377:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1267, %r1265, %r1266;
$L__tmp378:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1268, %r1160, 16, 31, -1;
$L__tmp379:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1269, %r1160, %r1268;
$L__tmp380:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1270, %r1269, 8, 31, -1;
$L__tmp381:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1271, %r1269, %r1270;
$L__tmp382:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1272, %r1271, 4, 31, -1;
$L__tmp383:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1273, %r1271, %r1272;
$L__tmp384:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1274, %r1273, 2, 31, -1;
$L__tmp385:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1275, %r1273, %r1274;
$L__tmp386:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1276, %r1275, 1, 31, -1;
$L__tmp387:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1277, %r1275, %r1276;
$L__tmp388:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1278, %r1163, 16, 31, -1;
$L__tmp389:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1279, %r1163, %r1278;
$L__tmp390:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1280, %r1279, 8, 31, -1;
$L__tmp391:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1281, %r1279, %r1280;
$L__tmp392:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1282, %r1281, 4, 31, -1;
$L__tmp393:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1283, %r1281, %r1282;
$L__tmp394:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1284, %r1283, 2, 31, -1;
$L__tmp395:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1285, %r1283, %r1284;
$L__tmp396:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1286, %r1285, 1, 31, -1;
$L__tmp397:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1287, %r1285, %r1286;
$L__tmp398:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1288, %r1166, 16, 31, -1;
$L__tmp399:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1289, %r1166, %r1288;
$L__tmp400:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1290, %r1289, 8, 31, -1;
$L__tmp401:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1291, %r1289, %r1290;
$L__tmp402:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1292, %r1291, 4, 31, -1;
$L__tmp403:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1293, %r1291, %r1292;
$L__tmp404:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1294, %r1293, 2, 31, -1;
$L__tmp405:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1295, %r1293, %r1294;
$L__tmp406:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1296, %r1295, 1, 31, -1;
$L__tmp407:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1297, %r1295, %r1296;
$L__tmp408:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1298, %r1169, 16, 31, -1;
$L__tmp409:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1299, %r1169, %r1298;
$L__tmp410:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1300, %r1299, 8, 31, -1;
$L__tmp411:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1301, %r1299, %r1300;
$L__tmp412:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1302, %r1301, 4, 31, -1;
$L__tmp413:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1303, %r1301, %r1302;
$L__tmp414:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1304, %r1303, 2, 31, -1;
$L__tmp415:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1305, %r1303, %r1304;
$L__tmp416:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1306, %r1305, 1, 31, -1;
$L__tmp417:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1307, %r1305, %r1306;
$L__tmp418:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1308, %r1172, 16, 31, -1;
$L__tmp419:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1309, %r1172, %r1308;
$L__tmp420:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1310, %r1309, 8, 31, -1;
$L__tmp421:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1311, %r1309, %r1310;
$L__tmp422:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1312, %r1311, 4, 31, -1;
$L__tmp423:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1313, %r1311, %r1312;
$L__tmp424:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1314, %r1313, 2, 31, -1;
$L__tmp425:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1315, %r1313, %r1314;
$L__tmp426:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1316, %r1315, 1, 31, -1;
$L__tmp427:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1317, %r1315, %r1316;
$L__tmp428:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1318, %r1175, 16, 31, -1;
$L__tmp429:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1319, %r1175, %r1318;
$L__tmp430:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1320, %r1319, 8, 31, -1;
$L__tmp431:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1321, %r1319, %r1320;
$L__tmp432:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1322, %r1321, 4, 31, -1;
$L__tmp433:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1323, %r1321, %r1322;
$L__tmp434:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1324, %r1323, 2, 31, -1;
$L__tmp435:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1325, %r1323, %r1324;
$L__tmp436:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1326, %r1325, 1, 31, -1;
$L__tmp437:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1327, %r1325, %r1326;
$L__tmp438:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1328, %r1178, 16, 31, -1;
$L__tmp439:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1329, %r1178, %r1328;
$L__tmp440:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1330, %r1329, 8, 31, -1;
$L__tmp441:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1331, %r1329, %r1330;
$L__tmp442:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1332, %r1331, 4, 31, -1;
$L__tmp443:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1333, %r1331, %r1332;
$L__tmp444:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1334, %r1333, 2, 31, -1;
$L__tmp445:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1335, %r1333, %r1334;
$L__tmp446:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1336, %r1335, 1, 31, -1;
$L__tmp447:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1337, %r1335, %r1336;
$L__tmp448:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1338, %r1181, 16, 31, -1;
$L__tmp449:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1339, %r1181, %r1338;
$L__tmp450:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1340, %r1339, 8, 31, -1;
$L__tmp451:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1341, %r1339, %r1340;
$L__tmp452:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1342, %r1341, 4, 31, -1;
$L__tmp453:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1343, %r1341, %r1342;
$L__tmp454:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1344, %r1343, 2, 31, -1;
$L__tmp455:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1345, %r1343, %r1344;
$L__tmp456:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1346, %r1345, 1, 31, -1;
$L__tmp457:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1347, %r1345, %r1346;
$L__tmp458:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1348, %r1184, 16, 31, -1;
$L__tmp459:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1349, %r1184, %r1348;
$L__tmp460:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1350, %r1349, 8, 31, -1;
$L__tmp461:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1351, %r1349, %r1350;
$L__tmp462:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1352, %r1351, 4, 31, -1;
$L__tmp463:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1353, %r1351, %r1352;
$L__tmp464:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1354, %r1353, 2, 31, -1;
$L__tmp465:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1355, %r1353, %r1354;
$L__tmp466:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1356, %r1355, 1, 31, -1;
$L__tmp467:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1357, %r1355, %r1356;
$L__tmp468:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1358, %r1187, 16, 31, -1;
$L__tmp469:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1359, %r1187, %r1358;
$L__tmp470:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1360, %r1359, 8, 31, -1;
$L__tmp471:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1361, %r1359, %r1360;
$L__tmp472:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1362, %r1361, 4, 31, -1;
$L__tmp473:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1363, %r1361, %r1362;
$L__tmp474:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1364, %r1363, 2, 31, -1;
$L__tmp475:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1365, %r1363, %r1364;
$L__tmp476:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1366, %r1365, 1, 31, -1;
$L__tmp477:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1367, %r1365, %r1366;
$L__tmp478:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1368, %r1190, 16, 31, -1;
$L__tmp479:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1369, %r1190, %r1368;
$L__tmp480:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1370, %r1369, 8, 31, -1;
$L__tmp481:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1371, %r1369, %r1370;
$L__tmp482:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1372, %r1371, 4, 31, -1;
$L__tmp483:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1373, %r1371, %r1372;
$L__tmp484:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1374, %r1373, 2, 31, -1;
$L__tmp485:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1375, %r1373, %r1374;
$L__tmp486:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1376, %r1375, 1, 31, -1;
$L__tmp487:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1377, %r1375, %r1376;
$L__tmp488:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1378, %r1193, 16, 31, -1;
$L__tmp489:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1379, %r1193, %r1378;
$L__tmp490:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1380, %r1379, 8, 31, -1;
$L__tmp491:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1381, %r1379, %r1380;
$L__tmp492:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1382, %r1381, 4, 31, -1;
$L__tmp493:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1383, %r1381, %r1382;
$L__tmp494:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1384, %r1383, 2, 31, -1;
$L__tmp495:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1385, %r1383, %r1384;
$L__tmp496:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1386, %r1385, 1, 31, -1;
$L__tmp497:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1387, %r1385, %r1386;
$L__tmp498:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1388, %r1196, 16, 31, -1;
$L__tmp499:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1389, %r1196, %r1388;
$L__tmp500:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1390, %r1389, 8, 31, -1;
$L__tmp501:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1391, %r1389, %r1390;
$L__tmp502:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1392, %r1391, 4, 31, -1;
$L__tmp503:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1393, %r1391, %r1392;
$L__tmp504:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1394, %r1393, 2, 31, -1;
$L__tmp505:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1395, %r1393, %r1394;
$L__tmp506:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1396, %r1395, 1, 31, -1;
$L__tmp507:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1397, %r1395, %r1396;
$L__tmp508:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1398, %r1199, 16, 31, -1;
$L__tmp509:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1399, %r1199, %r1398;
$L__tmp510:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1400, %r1399, 8, 31, -1;
$L__tmp511:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1401, %r1399, %r1400;
$L__tmp512:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1402, %r1401, 4, 31, -1;
$L__tmp513:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1403, %r1401, %r1402;
$L__tmp514:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1404, %r1403, 2, 31, -1;
$L__tmp515:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1405, %r1403, %r1404;
$L__tmp516:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1406, %r1405, 1, 31, -1;
$L__tmp517:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1407, %r1405, %r1406;
$L__tmp518:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1408, %r1202, 16, 31, -1;
$L__tmp519:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1409, %r1202, %r1408;
$L__tmp520:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1410, %r1409, 8, 31, -1;
$L__tmp521:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1411, %r1409, %r1410;
$L__tmp522:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1412, %r1411, 4, 31, -1;
$L__tmp523:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1413, %r1411, %r1412;
$L__tmp524:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1414, %r1413, 2, 31, -1;
$L__tmp525:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1415, %r1413, %r1414;
$L__tmp526:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1416, %r1415, 1, 31, -1;
$L__tmp527:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1417, %r1415, %r1416;
$L__tmp528:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1418, %r1205, 16, 31, -1;
$L__tmp529:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1419, %r1205, %r1418;
$L__tmp530:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1420, %r1419, 8, 31, -1;
$L__tmp531:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1421, %r1419, %r1420;
$L__tmp532:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1422, %r1421, 4, 31, -1;
$L__tmp533:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1423, %r1421, %r1422;
$L__tmp534:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1424, %r1423, 2, 31, -1;
$L__tmp535:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1425, %r1423, %r1424;
$L__tmp536:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1426, %r1425, 1, 31, -1;
$L__tmp537:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1427, %r1425, %r1426;
$L__tmp538:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1428, %r1208, 16, 31, -1;
$L__tmp539:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1429, %r1208, %r1428;
$L__tmp540:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1430, %r1429, 8, 31, -1;
$L__tmp541:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1431, %r1429, %r1430;
$L__tmp542:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1432, %r1431, 4, 31, -1;
$L__tmp543:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1433, %r1431, %r1432;
$L__tmp544:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1434, %r1433, 2, 31, -1;
$L__tmp545:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1435, %r1433, %r1434;
$L__tmp546:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1436, %r1435, 1, 31, -1;
$L__tmp547:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1437, %r1435, %r1436;
$L__tmp548:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1438, %r1211, 16, 31, -1;
$L__tmp549:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1439, %r1211, %r1438;
$L__tmp550:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1440, %r1439, 8, 31, -1;
$L__tmp551:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1441, %r1439, %r1440;
$L__tmp552:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1442, %r1441, 4, 31, -1;
$L__tmp553:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1443, %r1441, %r1442;
$L__tmp554:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1444, %r1443, 2, 31, -1;
$L__tmp555:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1445, %r1443, %r1444;
$L__tmp556:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1446, %r1445, 1, 31, -1;
$L__tmp557:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1447, %r1445, %r1446;
$L__tmp558:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1448, %r1214, 16, 31, -1;
$L__tmp559:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1449, %r1214, %r1448;
$L__tmp560:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1450, %r1449, 8, 31, -1;
$L__tmp561:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1451, %r1449, %r1450;
$L__tmp562:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1452, %r1451, 4, 31, -1;
$L__tmp563:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1453, %r1451, %r1452;
$L__tmp564:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1454, %r1453, 2, 31, -1;
$L__tmp565:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1455, %r1453, %r1454;
$L__tmp566:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1456, %r1455, 1, 31, -1;
$L__tmp567:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1457, %r1455, %r1456;
$L__tmp568:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1458, %r1217, 16, 31, -1;
$L__tmp569:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1459, %r1217, %r1458;
$L__tmp570:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1460, %r1459, 8, 31, -1;
$L__tmp571:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1461, %r1459, %r1460;
$L__tmp572:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1462, %r1461, 4, 31, -1;
$L__tmp573:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1463, %r1461, %r1462;
$L__tmp574:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1464, %r1463, 2, 31, -1;
$L__tmp575:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1465, %r1463, %r1464;
$L__tmp576:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1466, %r1465, 1, 31, -1;
$L__tmp577:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1467, %r1465, %r1466;
$L__tmp578:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1468, %r1220, 16, 31, -1;
$L__tmp579:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1469, %r1220, %r1468;
$L__tmp580:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1470, %r1469, 8, 31, -1;
$L__tmp581:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1471, %r1469, %r1470;
$L__tmp582:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1472, %r1471, 4, 31, -1;
$L__tmp583:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1473, %r1471, %r1472;
$L__tmp584:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1474, %r1473, 2, 31, -1;
$L__tmp585:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1475, %r1473, %r1474;
$L__tmp586:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1476, %r1475, 1, 31, -1;
$L__tmp587:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1477, %r1475, %r1476;
$L__tmp588:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1478, %r1223, 16, 31, -1;
$L__tmp589:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1479, %r1223, %r1478;
$L__tmp590:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1480, %r1479, 8, 31, -1;
$L__tmp591:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1481, %r1479, %r1480;
$L__tmp592:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1482, %r1481, 4, 31, -1;
$L__tmp593:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1483, %r1481, %r1482;
$L__tmp594:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1484, %r1483, 2, 31, -1;
$L__tmp595:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1485, %r1483, %r1484;
$L__tmp596:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1486, %r1485, 1, 31, -1;
$L__tmp597:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1487, %r1485, %r1486;
$L__tmp598:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1488, %r1226, 16, 31, -1;
$L__tmp599:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1489, %r1226, %r1488;
$L__tmp600:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1490, %r1489, 8, 31, -1;
$L__tmp601:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1491, %r1489, %r1490;
$L__tmp602:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1492, %r1491, 4, 31, -1;
$L__tmp603:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1493, %r1491, %r1492;
$L__tmp604:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1494, %r1493, 2, 31, -1;
$L__tmp605:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1495, %r1493, %r1494;
$L__tmp606:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1496, %r1495, 1, 31, -1;
$L__tmp607:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1497, %r1495, %r1496;
$L__tmp608:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1498, %r1229, 16, 31, -1;
$L__tmp609:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1499, %r1229, %r1498;
$L__tmp610:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1500, %r1499, 8, 31, -1;
$L__tmp611:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1501, %r1499, %r1500;
$L__tmp612:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1502, %r1501, 4, 31, -1;
$L__tmp613:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1503, %r1501, %r1502;
$L__tmp614:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1504, %r1503, 2, 31, -1;
$L__tmp615:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1505, %r1503, %r1504;
$L__tmp616:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1506, %r1505, 1, 31, -1;
$L__tmp617:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1507, %r1505, %r1506;
$L__tmp618:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1508, %r1232, 16, 31, -1;
$L__tmp619:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1509, %r1232, %r1508;
$L__tmp620:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1510, %r1509, 8, 31, -1;
$L__tmp621:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1511, %r1509, %r1510;
$L__tmp622:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1512, %r1511, 4, 31, -1;
$L__tmp623:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1513, %r1511, %r1512;
$L__tmp624:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1514, %r1513, 2, 31, -1;
$L__tmp625:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1515, %r1513, %r1514;
$L__tmp626:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1516, %r1515, 1, 31, -1;
$L__tmp627:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1517, %r1515, %r1516;
$L__tmp628:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1518, %r1235, 16, 31, -1;
$L__tmp629:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1519, %r1235, %r1518;
$L__tmp630:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1520, %r1519, 8, 31, -1;
$L__tmp631:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1521, %r1519, %r1520;
$L__tmp632:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1522, %r1521, 4, 31, -1;
$L__tmp633:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1523, %r1521, %r1522;
$L__tmp634:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1524, %r1523, 2, 31, -1;
$L__tmp635:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1525, %r1523, %r1524;
$L__tmp636:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1526, %r1525, 1, 31, -1;
$L__tmp637:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1527, %r1525, %r1526;
$L__tmp638:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1528, %r1238, 16, 31, -1;
$L__tmp639:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1529, %r1238, %r1528;
$L__tmp640:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1530, %r1529, 8, 31, -1;
$L__tmp641:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1531, %r1529, %r1530;
$L__tmp642:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1532, %r1531, 4, 31, -1;
$L__tmp643:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1533, %r1531, %r1532;
$L__tmp644:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1534, %r1533, 2, 31, -1;
$L__tmp645:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1535, %r1533, %r1534;
$L__tmp646:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1536, %r1535, 1, 31, -1;
$L__tmp647:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1537, %r1535, %r1536;
$L__tmp648:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1538, %r1241, 16, 31, -1;
$L__tmp649:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1539, %r1241, %r1538;
$L__tmp650:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1540, %r1539, 8, 31, -1;
$L__tmp651:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1541, %r1539, %r1540;
$L__tmp652:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1542, %r1541, 4, 31, -1;
$L__tmp653:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1543, %r1541, %r1542;
$L__tmp654:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1544, %r1543, 2, 31, -1;
$L__tmp655:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1545, %r1543, %r1544;
$L__tmp656:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1546, %r1545, 1, 31, -1;
$L__tmp657:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1547, %r1545, %r1546;
$L__tmp658:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1548, %r1244, 16, 31, -1;
$L__tmp659:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1549, %r1244, %r1548;
$L__tmp660:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1550, %r1549, 8, 31, -1;
$L__tmp661:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1551, %r1549, %r1550;
$L__tmp662:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1552, %r1551, 4, 31, -1;
$L__tmp663:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1553, %r1551, %r1552;
$L__tmp664:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1554, %r1553, 2, 31, -1;
$L__tmp665:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1555, %r1553, %r1554;
$L__tmp666:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1556, %r1555, 1, 31, -1;
$L__tmp667:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1557, %r1555, %r1556;
$L__tmp668:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1558, %r1247, 16, 31, -1;
$L__tmp669:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1559, %r1247, %r1558;
$L__tmp670:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1560, %r1559, 8, 31, -1;
$L__tmp671:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1561, %r1559, %r1560;
$L__tmp672:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1562, %r1561, 4, 31, -1;
$L__tmp673:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1563, %r1561, %r1562;
$L__tmp674:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1564, %r1563, 2, 31, -1;
$L__tmp675:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1565, %r1563, %r1564;
$L__tmp676:
	.loc	2 293 36                        // standard.py:293:36 @[ sk08_ssm_control.py:152:17 ]
	shfl.sync.bfly.b32 	%r1566, %r1565, 1, 31, -1;
$L__tmp677:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk08_ssm_control.py:152:17 ] ]
	add.f32 	%r1567, %r1565, %r1566;
$L__tmp678:
	.loc	1 154 22                        // sk08_ssm_control.py:154:22
	// begin inline asm
	st.global.v4.b32 [ %rd30 + 0 ], { %r266, %r267, %r268, %r269 };
	// end inline asm
	// begin inline asm
	st.global.v4.b32 [ %rd31 + 0 ], { %r270, %r271, %r272, %r273 };
	// end inline asm
	// begin inline asm
	st.global.v4.b32 [ %rd32 + 0 ], { %r274, %r275, %r276, %r277 };
	// end inline asm
	// begin inline asm
	st.global.v4.b32 [ %rd33 + 0 ], { %r278, %r279, %r280, %r281 };
	// end inline asm
	// begin inline asm
	st.global.v4.b32 [ %rd34 + 0 ], { %r282, %r283, %r284, %r285 };
	// end inline asm
	// begin inline asm
	st.global.v4.b32 [ %rd35 + 0 ], { %r286, %r287, %r288, %r289 };
	// end inline asm
	// begin inline asm
	st.global.v4.b32 [ %rd36 + 0 ], { %r290, %r291, %r292, %r293 };
	// end inline asm
	// begin inline asm
	st.global.v4.b32 [ %rd37 + 0 ], { %r294, %r295, %r296, %r297 };
	// end inline asm
	// begin inline asm
	st.global.v4.b32 [ %rd38 + 0 ], { %r298, %r299, %r300, %r301 };
	// end inline asm
	// begin inline asm
	st.global.v4.b32 [ %rd39 + 0 ], { %r302, %r303, %r304, %r305 };
	// end inline asm
	// begin inline asm
	st.global.v4.b32 [ %rd40 + 0 ], { %r306, %r307, %r308, %r309 };
	// end inline asm
	// begin inline asm
	st.global.v4.b32 [ %rd41 + 0 ], { %r310, %r311, %r312, %r313 };
	// end inline asm
	// begin inline asm
	st.global.v4.b32 [ %rd42 + 0 ], { %r314, %r315, %r316, %r317 };
	// end inline asm
	// begin inline asm
	st.global.v4.b32 [ %rd43 + 0 ], { %r318, %r319, %r320, %r321 };
	// end inline asm
	// begin inline asm
	st.global.v4.b32 [ %rd44 + 0 ], { %r322, %r323, %r324, %r325 };
	// end inline asm
	// begin inline asm
	st.global.v4.b32 [ %rd45 + 0 ], { %r326, %r327, %r328, %r329 };
	// end inline asm
	// begin inline asm
	st.global.v4.b32 [ %rd46 + 0 ], { %r330, %r331, %r332, %r333 };
	// end inline asm
	// begin inline asm
	st.global.v4.b32 [ %rd47 + 0 ], { %r334, %r335, %r336, %r337 };
	// end inline asm
	// begin inline asm
	st.global.v4.b32 [ %rd48 + 0 ], { %r338, %r339, %r340, %r341 };
	// end inline asm
	// begin inline asm
	st.global.v4.b32 [ %rd49 + 0 ], { %r342, %r343, %r344, %r345 };
	// end inline asm
	// begin inline asm
	st.global.v4.b32 [ %rd50 + 0 ], { %r346, %r347, %r348, %r349 };
	// end inline asm
	// begin inline asm
	st.global.v4.b32 [ %rd51 + 0 ], { %r350, %r351, %r352, %r353 };
	// end inline asm
	// begin inline asm
	st.global.v4.b32 [ %rd52 + 0 ], { %r354, %r355, %r356, %r357 };
	// end inline asm
	// begin inline asm
	st.global.v4.b32 [ %rd53 + 0 ], { %r358, %r359, %r360, %r361 };
	// end inline asm
	// begin inline asm
	st.global.v4.b32 [ %rd54 + 0 ], { %r362, %r363, %r364, %r365 };
	// end inline asm
	// begin inline asm
	st.global.v4.b32 [ %rd55 + 0 ], { %r366, %r367, %r368, %r369 };
	// end inline asm
	// begin inline asm
	st.global.v4.b32 [ %rd56 + 0 ], { %r370, %r371, %r372, %r373 };
	// end inline asm
	// begin inline asm
	st.global.v4.b32 [ %rd57 + 0 ], { %r374, %r375, %r376, %r377 };
	// end inline asm
	// begin inline asm
	st.global.v4.b32 [ %rd58 + 0 ], { %r378, %r379, %r380, %r381 };
	// end inline asm
	// begin inline asm
	st.global.v4.b32 [ %rd59 + 0 ], { %r382, %r383, %r384, %r385 };
	// end inline asm
	// begin inline asm
	st.global.v4.b32 [ %rd60 + 0 ], { %r386, %r387, %r388, %r389 };
	// end inline asm
	// begin inline asm
	st.global.v4.b32 [ %rd61 + 0 ], { %r390, %r391, %r392, %r393 };
	// end inline asm
	.loc	1 155 27                        // sk08_ssm_control.py:155:27
	mul.lo.s32 	%r1568, %r29, %r1;
	.loc	1 155 21                        // sk08_ssm_control.py:155:21
	mad.wide.s32 	%rd64, %r1568, 2, %rd5;
	.loc	1 155 40                        // sk08_ssm_control.py:155:40
	shl.b64 	%rd65, %rd2, 1;
	add.s64 	%rd66, %rd64, %rd65;
	.loc	1 155 51                        // sk08_ssm_control.py:155:51
	shl.b64 	%rd67, %rd1, 1;
	add.s64 	%rd62, %rd66, %rd67;
	.loc	1 155 63                        // sk08_ssm_control.py:155:63
	cvt.rn.bf16x2.f32 	%r395, %r1337, %r1257;
	cvt.rn.bf16x2.f32 	%r396, %r1347, %r1267;
	cvt.rn.bf16x2.f32 	%r397, %r1357, %r1277;
	cvt.rn.bf16x2.f32 	%r398, %r1367, %r1287;
	cvt.rn.bf16x2.f32 	%r405, %r1497, %r1417;
	cvt.rn.bf16x2.f32 	%r406, %r1507, %r1427;
	cvt.rn.bf16x2.f32 	%r407, %r1517, %r1437;
	cvt.rn.bf16x2.f32 	%r408, %r1527, %r1447;
	.loc	1 155 56                        // sk08_ssm_control.py:155:56
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r394 + 0 ], { %r395, %r396, %r397, %r398 };
	// end inline asm
	cvt.rn.bf16x2.f32 	%r400, %r1377, %r1297;
	cvt.rn.bf16x2.f32 	%r401, %r1387, %r1307;
	cvt.rn.bf16x2.f32 	%r402, %r1397, %r1317;
	cvt.rn.bf16x2.f32 	%r403, %r1407, %r1327;
	// begin inline asm
	st.shared.v4.b32 [ %r399 + 0 ], { %r400, %r401, %r402, %r403 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r404 + 0 ], { %r405, %r406, %r407, %r408 };
	// end inline asm
	cvt.rn.bf16x2.f32 	%r410, %r1537, %r1457;
	cvt.rn.bf16x2.f32 	%r411, %r1547, %r1467;
	cvt.rn.bf16x2.f32 	%r412, %r1557, %r1477;
	cvt.rn.bf16x2.f32 	%r413, %r1567, %r1487;
	// begin inline asm
	st.shared.v4.b32 [ %r409 + 0 ], { %r410, %r411, %r412, %r413 };
	// end inline asm
	bar.sync 	0;
	ld.shared.b16 	%rs8, [%r8];
	// begin inline asm
	st.global.b16 [ %rd62 + 0 ], { %rs8 };
	// end inline asm
	.loc	1 155 4                         // sk08_ssm_control.py:155:4
	ret;
$L__tmp679:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk08_ssm_control.py"
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
.b8 1                                   // DW_CHILDREN_yes
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
.b8 5                                   // Abbreviation Code
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
.b8 5                                   // DW_FORM_data2
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 6                                   // Abbreviation Code
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
.b32 348                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x155 DW_TAG_compile_unit
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
.b8 48
.b8 56
.b8 95
.b8 115
.b8 115
.b8 109
.b8 95
.b8 99
.b8 111
.b8 110
.b8 116
.b8 114
.b8 111
.b8 108
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
.b8 2                                   // Abbrev [2] 0x49:0x20 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 56
.b8 95
.b8 103
.b8 97
.b8 116
.b8 101
.b8 100
.b8 95
.b8 100
.b8 101
.b8 108
.b8 116
.b8 97
.b8 95
.b8 114
.b8 117
.b8 108
.b8 101
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x69:0xf6 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 73                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x7e:0x32 DW_TAG_inlined_subroutine
.b32 73                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp16                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 138                                 // DW_AT_call_line
.b8 32                                  // DW_AT_call_column
.b8 5                                   // Abbrev [5] 0x96:0x19 DW_TAG_inlined_subroutine
.b32 73                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp15                          // DW_AT_high_pc
.b8 2                                   // DW_AT_call_file
.b8 37                                  // DW_AT_call_line
.b8 1
.b8 36                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 4                                   // Abbrev [4] 0xb0:0x32 DW_TAG_inlined_subroutine
.b32 73                                 // DW_AT_abstract_origin
.b64 $L__tmp17                          // DW_AT_low_pc
.b64 $L__tmp32                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 139                                 // DW_AT_call_line
.b8 32                                  // DW_AT_call_column
.b8 5                                   // Abbrev [5] 0xc8:0x19 DW_TAG_inlined_subroutine
.b32 73                                 // DW_AT_abstract_origin
.b64 $L__tmp18                          // DW_AT_low_pc
.b64 $L__tmp31                          // DW_AT_high_pc
.b8 2                                   // DW_AT_call_file
.b8 37                                  // DW_AT_call_line
.b8 1
.b8 36                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 6                                   // Abbrev [6] 0xe2:0x18 DW_TAG_inlined_subroutine
.b32 73                                 // DW_AT_abstract_origin
.b64 $L__tmp33                          // DW_AT_low_pc
.b64 $L__tmp34                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 146                                 // DW_AT_call_line
.b8 26                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0xfa:0x32 DW_TAG_inlined_subroutine
.b32 73                                 // DW_AT_abstract_origin
.b64 $L__tmp35                          // DW_AT_low_pc
.b64 $L__tmp356                         // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 149                                 // DW_AT_call_line
.b8 18                                  // DW_AT_call_column
.b8 5                                   // Abbrev [5] 0x112:0x19 DW_TAG_inlined_subroutine
.b32 73                                 // DW_AT_abstract_origin
.b64 $L__tmp35                          // DW_AT_low_pc
.b64 $L__tmp356                         // DW_AT_high_pc
.b8 2                                   // DW_AT_call_file
.b8 37                                  // DW_AT_call_line
.b8 1
.b8 36                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 4                                   // Abbrev [4] 0x12c:0x32 DW_TAG_inlined_subroutine
.b32 73                                 // DW_AT_abstract_origin
.b64 $L__tmp357                         // DW_AT_low_pc
.b64 $L__tmp678                         // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 152                                 // DW_AT_call_line
.b8 17                                  // DW_AT_call_column
.b8 5                                   // Abbrev [5] 0x144:0x19 DW_TAG_inlined_subroutine
.b32 73                                 // DW_AT_abstract_origin
.b64 $L__tmp357                         // DW_AT_low_pc
.b64 $L__tmp678                         // DW_AT_high_pc
.b8 2                                   // DW_AT_call_file
.b8 37                                  // DW_AT_call_line
.b8 1
.b8 36                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_QVAR1 = _Nativo(
    "sk08_ssm_control/_sk08_gated_delta_rule_kernel",
    _QPTX1, "_sk08_gated_delta_rule_kernel",
    warps=4, shared=512,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    horneado={11: 1.0, 12: 16, 13: 48, 14: 128, 15: 128, 16: 20.0},
    div16=[7, 8, 9, 10],
)


def _q1_impl(grid: list[int], qkv_ptr: torch.Tensor, a_ptr: torch.Tensor, b_ptr: torch.Tensor, A_log_ptr: torch.Tensor, dt_bias_ptr: torch.Tensor, state_ptr: torch.Tensor, o_ptr: torch.Tensor, stride_qkv_b: int, stride_a_b: int, stride_state_b: int, stride_o_b: int, SCALE: float, H: int, HV: int, K: int, V: int, SOFTPLUS_THRESHOLD: float) -> None:
    """Cuerpo del custom op: lanza el PTX embebido."""
    _QVAR1(tuple(grid), qkv_ptr, a_ptr, b_ptr, A_log_ptr, dt_bias_ptr, state_ptr, o_ptr, stride_qkv_b, stride_a_b, stride_state_b, stride_o_b, SCALE, H, HV, K, V, SOFTPLUS_THRESHOLD)


def _q1_fake(grid: list[int], qkv_ptr: torch.Tensor, a_ptr: torch.Tensor, b_ptr: torch.Tensor, A_log_ptr: torch.Tensor, dt_bias_ptr: torch.Tensor, state_ptr: torch.Tensor, o_ptr: torch.Tensor, stride_qkv_b: int, stride_a_b: int, stride_state_b: int, stride_o_b: int, SCALE: float, H: int, HV: int, K: int, V: int, SOFTPLUS_THRESHOLD: float) -> None:
    """Meta impl: no toca la GPU; las salidas se mutan in-place."""
    return None


_q1_registrado = False


def _q1_registrar() -> None:
    """Registra el op una sola vez, fuera de la region compilada."""
    global _q1_registrado
    if _q1_registrado:
        return
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(
        op_name="genesis_sk08_ssm_control_q1",
        op_func=_q1_impl,
        mutates_args=['state_ptr', 'o_ptr'],
        fake_impl=_q1_fake,
    )
    _q1_registrado = True


def _lanzar_quant1(grid, *args):
    """PTX embebido via custom op; cae al Triton con el kill-switch.

    ``GENESIS_PTQ_NATIVO=0`` vuelve al kernel Triton, que queda en este
    archivo como referencia para los tests.
    """
    if habilitado():
        _q1_registrar()
        g = [grid] if isinstance(grid, int) else list(grid)
        return torch.ops.vllm.genesis_sk08_ssm_control_q1(g, *args)
    ce = ['SCALE', 'H', 'HV', 'K', 'V', 'SOFTPLUS_THRESHOLD']
    nom = ['qkv_ptr', 'a_ptr', 'b_ptr', 'A_log_ptr', 'dt_bias_ptr', 'state_ptr', 'o_ptr', 'stride_qkv_b', 'stride_a_b', 'stride_state_b', 'stride_o_b', 'SCALE', 'H', 'HV', 'K', 'V', 'SOFTPLUS_THRESHOLD']
    return _sk08_gated_delta_rule_kernel[grid](*args[:len(nom) - len(ce)],
                    **{n: args[nom.index(n)] for n in ce},
                    num_warps=8, num_stages=1)
