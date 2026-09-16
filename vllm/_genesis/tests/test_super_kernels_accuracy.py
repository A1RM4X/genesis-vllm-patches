#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""test_super_kernels_accuracy.py — Suite de pruebas de precisión y contratos de tensores para Super-Kernels sm_86.

Valida numéricamente cada Super-Kernel contra pesos reales de safetensors y referencia FP32
descuantizada a través de todo el espectro de tokens (M = 1, 4, 8, 16, 32, 64, 69, 128, 512, 1024):
- SK-01: GDN QKVZ FUSED INT8
- SK-02: GDN OUT INT8 SCALED
- SK-03: FA QKV FUSED INT8
- SK-04: FA O INT8 SCALED
- SK-05: MLP GATEUP FUSED INT8
- SK-06: MLP DOWN INT8 SCALED
- SK-07: LM HEAD INT8 SCALED
- SK-10: MTP DRAFT FUSED INT8
"""

import os
import pytest
import torch
from safetensors import safe_open

from vllm._genesis.wiring.quantization.patch_PN110_int8_phase_dispatch import _build_int8_state
from vllm._genesis.kernels.sk01_gdn_qkvz import sk01_gdn_qkvz_gemm
from vllm._genesis.kernels.sk02_gdn_out import sk02_gemm_int8_scaled
from vllm._genesis.kernels.sk03_fa_qkv import sk03_fa_qkv_gemm
from vllm._genesis.kernels.sk04_fa_o import fa_o_int8_scaled_gemm
from vllm._genesis.kernels.sk05_mlp_gateup import sk05_gateup_gemm
from vllm._genesis.kernels.sk06_mlp_down import mlp_down_gemm
from vllm._genesis.kernels.sk07_lm_head import lm_head_gemm, _arange
from vllm._genesis.kernels.sk09_norm_embed import quant_per_token
from vllm._genesis.kernels.sk10_mtp_draft import mtp_draft_fused_gemm


class DummyLayer:
    def __init__(self, w, s, k, n, bias=None):
        self.weight = w
        self.weight_scale_inv = s
        self.input_size_per_partition = k
        self.output_size_per_partition = n
        self.bias = bias


def get_safetensors_shard():
    snap_dir = "/root/.cache/huggingface/hub/models--orcarouter--Qwen3.8-27B-Uncensored-FP8/snapshots"
    if not os.path.exists(snap_dir):
        return None
    sub = os.listdir(snap_dir)
    if not sub:
        return None
    snap_dir = os.path.join(snap_dir, sub[0])
    f_path = os.path.join(snap_dir, "model-00001-of-00007.safetensors")
    if os.path.exists(f_path):
        return f_path
    return None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requiere GPU CUDA sm_86")
class TestSuperKernelsAccuracy:

    @classmethod
    def setup_class(cls):
        cls.shard_path = get_safetensors_shard()
        cls.device = "cuda:0"

    def _ref_int8(self, a_i8, a_sc, b_col, bsc, shifts=None, shift_block=128):
        """Lo que el kernel TIENE que calcular, en torch puro.

        Comparar contra ``x.float() @ w_fp32.t()`` mezcla dos cosas: si el
        kernel esta bien, y cuanto pierde el esquema de cuantizacion. Y lo
        segundo domina: ``_build_int8_state`` colapsa las escalas por bloque de
        128 a lo largo de K en UNA escala por columna, asi que con las escalas
        sinteticas de estos tests (``rand*0.01 + 1e-3``, un rango de 10x entre
        bloques) el camino int8 IDEAL ya da cos=0.902 contra fp32 — medido, y
        es exactamente lo que devuelve el kernel. En el modelo real el rango
        entre bloques es 1,7-2,0x, o sea mucho mas benigno.

        Asi que la correctitud del kernel se mide contra esta referencia, y la
        perdida del esquema se chequea aparte con ``_cota_esquema``.
        """
        B = b_col.float()
        if B.shape[0] != a_i8.shape[1]:
            B = B.t()
        if shifts is not None and bool(shifts.any()):
            # Camino HAS_SHIFT: el kernel acumula por bloque de K haciendo
            # `acc += d * shift[kb, n // SHIFT_BLOCK]` y NO aplica b_scales
            # (esa rama es el `if not HAS_SHIFT` del epilogo). El corrimiento
            # diadico recupera por bloque la escala que la escala por columna
            # habia colapsado, asi que ignorarlo da 0,90 contra un kernel sano.
            K = a_i8.shape[1]
            acc = torch.zeros((a_i8.shape[0], B.shape[1]),
                              dtype=torch.float32, device=a_i8.device)
            for kb in range(K // shift_block):
                sl = slice(kb * shift_block, (kb + 1) * shift_block)
                d = a_i8[:, sl].float() @ B[sl, :]
                sh = shifts[kb].float()
                acc += d * sh.repeat_interleave(B.shape[1] // sh.numel())[None, :]
            return acc * a_sc.reshape(-1, 1).float()
        acc = a_i8.float() @ B
        return acc * a_sc.reshape(-1, 1).float() * bsc.float().reshape(1, -1)

    def _cota_esquema(self, ref_int8, ref_fp32):
        """Perdida del esquema de cuantizacion, informativa, no del kernel."""
        return torch.nn.functional.cosine_similarity(
            ref_int8.reshape(-1).float(), ref_fp32.reshape(-1).float(), dim=0).item()

    def _dequant_weight(self, w, s, K, N, bk=128, bn=128):
        w_fp32 = torch.empty((N, K), dtype=torch.float32, device=self.device)
        for r in range(0, N, bn):
            for c in range(0, K, bk):
                w_fp32[r:r+bn, c:c+bk] = w[r:r+bn, c:c+bk].to(torch.float32) * s[r//bn, c//bk].to(torch.float32)
        return w_fp32

    @pytest.mark.parametrize("M", [1, 4, 8, 16, 32, 64, 69, 128, 512, 1024])
    def test_sk01_gdn_qkvz_accuracy(self, M):
        """Valida SK-01 (GDN QKVZ) contra referencia FP32."""
        K, N = 5120, 8192
        torch.manual_seed(42)
        w = torch.randint(-127, 127, (N, K), dtype=torch.int8, device=self.device).to(torch.float8_e4m3fn)
        s = torch.rand((N // 128, K // 128), dtype=torch.bfloat16, device=self.device) * 0.01 + 1e-3
        w_fp32 = self._dequant_weight(w, s, K, N)
        layer = DummyLayer(w, s, K, N)
        state = _build_int8_state(layer, w, s, (128, 128))
        bsc = state["b_scales"].reshape(-1)
        shifts = torch.zeros((K // 128, N // 128), dtype=torch.float32, device=self.device)

        x = torch.randn((M, K), dtype=torch.bfloat16, device=self.device)
        a_i8, a_sc = quant_per_token(x)
        out_sk = sk01_gdn_qkvz_gemm(a_i8, state["b_col"], a_sc.reshape(-1), bsc, shifts, None, torch.bfloat16)
        out_ref = self._ref_int8(a_i8, a_sc, state["b_col"], bsc)
        assert self._cota_esquema(out_ref, x.float() @ w_fp32.t()) > 0.5

        cos_sim = torch.nn.functional.cosine_similarity(out_sk.float().reshape(-1), out_ref.float().reshape(-1), dim=0).item()
        assert cos_sim >= 0.9995, f"SK-01 cos_sim={cos_sim:.6f} falló para M={M}"

    @pytest.mark.parametrize("M", [1, 4, 8, 16, 32, 64, 69, 128, 512, 1024])
    def test_sk02_gdn_out_accuracy(self, M):
        """Valida SK-02 (GDN Out) contra referencia FP32."""
        K, N = 5120, 5120
        torch.manual_seed(42)
        w = torch.randint(-127, 127, (N, K), dtype=torch.int8, device=self.device).to(torch.float8_e4m3fn)
        s = torch.rand((N // 128, K // 128), dtype=torch.bfloat16, device=self.device) * 0.01 + 1e-3
        w_fp32 = self._dequant_weight(w, s, K, N)
        layer = DummyLayer(w, s, K, N)
        state = _build_int8_state(layer, w, s, (128, 128))
        bsc = state["b_scales"].reshape(-1)
        shifts = torch.zeros((K // 128, N // 128), dtype=torch.float32, device=self.device)

        x = torch.randn((M, K), dtype=torch.bfloat16, device=self.device)
        a_i8, a_sc = quant_per_token(x)
        out_sk = sk02_gemm_int8_scaled(a_i8, state["b_col"], a_sc.reshape(-1), bsc, shifts, None, torch.bfloat16)
        out_ref = self._ref_int8(a_i8, a_sc, state["b_col"], bsc)
        assert self._cota_esquema(out_ref, x.float() @ w_fp32.t()) > 0.5

        cos_sim = torch.nn.functional.cosine_similarity(out_sk.float().reshape(-1), out_ref.float().reshape(-1), dim=0).item()
        assert cos_sim >= 0.9995, f"SK-02 cos_sim={cos_sim:.6f} falló para M={M}"

    @pytest.mark.parametrize("M", [1, 4, 8, 16, 32, 64, 69, 128, 512, 1024])
    def test_sk03_fa_qkv_accuracy(self, M):
        """Valida SK-03 (Full Attention QKV) contra referencia FP32."""
        K, N = 5120, 7168
        torch.manual_seed(42)
        if self.shard_path:
            with safe_open(self.shard_path, framework="pt", device=self.device) as f:
                wq = f.get_tensor("model.language_model.layers.3.self_attn.q_proj.weight")[:6144, :]
                wk = f.get_tensor("model.language_model.layers.3.self_attn.k_proj.weight")[:512, :]
                wv = f.get_tensor("model.language_model.layers.3.self_attn.v_proj.weight")[:512, :]
                w = torch.cat([wq, wk, wv], dim=0).contiguous()
                sq = f.get_tensor("model.language_model.layers.3.self_attn.q_proj.weight_scale_inv")[:48, :]
                sk = f.get_tensor("model.language_model.layers.3.self_attn.k_proj.weight_scale_inv")[:4, :]
                sv = f.get_tensor("model.language_model.layers.3.self_attn.v_proj.weight_scale_inv")[:4, :]
                s = torch.cat([sq, sk, sv], dim=0).contiguous()
        else:
            w = torch.randint(-127, 127, (N, K), dtype=torch.int8, device=self.device).to(torch.float8_e4m3fn)
            s = torch.rand((N // 128, K // 128), dtype=torch.bfloat16, device=self.device) * 0.01 + 1e-3

        w_fp32 = self._dequant_weight(w, s, K, N)
        layer = DummyLayer(w, s, K, N)
        state = _build_int8_state(layer, w, s, (128, 128))
        bsc = state["b_scales"].reshape(-1)
        shifts = torch.zeros((K // 128, N // 128), dtype=torch.float32, device=self.device)

        x = torch.randn((M, K), dtype=torch.bfloat16, device=self.device)
        a_i8, a_sc = quant_per_token(x)
        out_sk = sk03_fa_qkv_gemm(a_i8, state["b_col"], a_sc.reshape(-1), bsc, shifts, None, torch.bfloat16)
        out_ref = self._ref_int8(a_i8, a_sc, state["b_col"], bsc)
        assert self._cota_esquema(out_ref, x.float() @ w_fp32.t()) > 0.5

        cos_sim = torch.nn.functional.cosine_similarity(out_sk.float().reshape(-1), out_ref.float().reshape(-1), dim=0).item()
        assert cos_sim >= 0.9995, f"SK-03 cos_sim={cos_sim:.6f} falló para M={M}"

    @pytest.mark.parametrize("M", [1, 4, 8, 16, 32, 64, 69, 128, 512, 1024])
    def test_sk04_fa_o_accuracy(self, M):
        """Valida SK-04 (Full Attention Out) contra referencia FP32."""
        K, N = 3072, 5120
        torch.manual_seed(42)
        if self.shard_path:
            with safe_open(self.shard_path, framework="pt", device=self.device) as f:
                w = f.get_tensor("model.language_model.layers.3.self_attn.o_proj.weight")[:, :3072].contiguous()
                s = f.get_tensor("model.language_model.layers.3.self_attn.o_proj.weight_scale_inv")[:, :24].contiguous()
        else:
            w = torch.randint(-127, 127, (N, K), dtype=torch.int8, device=self.device).to(torch.float8_e4m3fn)
            s = torch.rand((N // 128, K // 128), dtype=torch.bfloat16, device=self.device) * 0.01 + 1e-3

        w_fp32 = self._dequant_weight(w, s, K, N)
        layer = DummyLayer(w, s, K, N)
        state = _build_int8_state(layer, w, s, (128, 128))
        bsc = state["b_scales"].reshape(-1)
        shifts = torch.zeros((K // 128, N // 128), dtype=torch.float32, device=self.device)

        x = torch.randn((M, K), dtype=torch.bfloat16, device=self.device)
        a_i8, a_sc = quant_per_token(x)
        out_sk = fa_o_int8_scaled_gemm(a_i8, state["b_col"], a_sc.reshape(-1), bsc, shifts, None, torch.bfloat16)
        out_ref = self._ref_int8(a_i8, a_sc, state["b_col"], bsc)
        assert self._cota_esquema(out_ref, x.float() @ w_fp32.t()) > 0.5

        cos_sim = torch.nn.functional.cosine_similarity(out_sk.float().reshape(-1), out_ref.float().reshape(-1), dim=0).item()
        assert cos_sim >= 0.9995, f"SK-04 cos_sim={cos_sim:.6f} falló para M={M}"

    @pytest.mark.parametrize("M", [1, 4, 8, 16, 32, 64, 69, 128, 512, 1024])
    def test_sk05_gateup_accuracy(self, M):
        """Valida SK-05 (MLP GateUp) contra referencia FP32."""
        K, N = 5120, 17408
        torch.manual_seed(42)
        if self.shard_path:
            with safe_open(self.shard_path, framework="pt", device=self.device) as f:
                wg = f.get_tensor("model.language_model.layers.0.mlp.gate_proj.weight")[:8704, :]
                wu = f.get_tensor("model.language_model.layers.0.mlp.up_proj.weight")[:8704, :]
                w = torch.cat([wg, wu], dim=0).contiguous()
                sg = f.get_tensor("model.language_model.layers.0.mlp.gate_proj.weight_scale_inv")[:68, :]
                su = f.get_tensor("model.language_model.layers.0.mlp.up_proj.weight_scale_inv")[:68, :]
                s = torch.cat([sg, su], dim=0).contiguous()
        else:
            w = torch.randint(-127, 127, (N, K), dtype=torch.int8, device=self.device).to(torch.float8_e4m3fn)
            s = torch.rand((N // 128, K // 128), dtype=torch.bfloat16, device=self.device) * 0.01 + 1e-3

        w_fp32 = self._dequant_weight(w, s, K, N)
        layer = DummyLayer(w, s, K, N)
        state = _build_int8_state(layer, w, s, (128, 128))
        bsc = state["b_scales"].reshape(-1)
        shifts = torch.zeros((K // 128, N // 128), dtype=torch.float32, device=self.device)

        x = torch.randn((M, K), dtype=torch.bfloat16, device=self.device)
        a_i8, a_sc = quant_per_token(x)
        out_sk = sk05_gateup_gemm(a_i8, state["b_col"], a_sc.reshape(-1), bsc, shifts, None, torch.bfloat16)
        out_ref = self._ref_int8(a_i8, a_sc, state["b_col"], bsc)
        assert self._cota_esquema(out_ref, x.float() @ w_fp32.t()) > 0.5

        cos_sim = torch.nn.functional.cosine_similarity(out_sk.float().reshape(-1), out_ref.float().reshape(-1), dim=0).item()
        assert cos_sim >= 0.9995, f"SK-05 cos_sim={cos_sim:.6f} falló para M={M}"

    @pytest.mark.parametrize("M", [1, 4, 8, 16, 32, 64, 69, 128, 512, 1024])
    def test_sk06_down_accuracy(self, M):
        """Valida SK-06 (MLP Down) contra referencia FP32."""
        K, N = 8704, 5120
        torch.manual_seed(42)
        if self.shard_path:
            with safe_open(self.shard_path, framework="pt", device=self.device) as f:
                w = f.get_tensor("model.language_model.layers.0.mlp.down_proj.weight")[:, :8704].contiguous()
                s = f.get_tensor("model.language_model.layers.0.mlp.down_proj.weight_scale_inv")[:, :68].contiguous()
        else:
            w = torch.randint(-127, 127, (N, K), dtype=torch.int8, device=self.device).to(torch.float8_e4m3fn)
            s = torch.rand((N // 128, K // 128), dtype=torch.bfloat16, device=self.device) * 0.01 + 1e-3

        w_fp32 = self._dequant_weight(w, s, K, N)
        layer = DummyLayer(w, s, K, N)
        state = _build_int8_state(layer, w, s, (128, 128))
        bsc = state["b_scales"].reshape(-1)
        shifts = state.get("w_shifts")
        if shifts is None:
            shifts = torch.zeros((K // 128, N // 128), dtype=torch.float32, device=self.device)

        x = torch.randn((M, K), dtype=torch.bfloat16, device=self.device)
        a_i8, a_sc = quant_per_token(x)
        out_sk = mlp_down_gemm(a_i8, state["b_col"], a_sc.reshape(-1), bsc, shifts, None, torch.bfloat16)
        out_ref = self._ref_int8(a_i8, a_sc, state["b_col"], bsc, shifts)
        assert self._cota_esquema(out_ref, x.float() @ w_fp32.t()) > 0.5

        cos_sim = torch.nn.functional.cosine_similarity(out_sk.float().reshape(-1), out_ref.float().reshape(-1), dim=0).item()
        assert cos_sim >= 0.9995, f"SK-06 cos_sim={cos_sim:.6f} falló para M={M}"

    @pytest.mark.parametrize("M", [1, 4, 8, 16, 32, 64, 128])
    def test_sk07_lm_head_accuracy(self, M):
        """Valida SK-07 (LM Head) contra referencia FP32."""
        K, N = 5120, 152064
        torch.manual_seed(42)
        w = torch.randint(-127, 127, (N, K), dtype=torch.int8, device=self.device).to(torch.float8_e4m3fn)
        s = torch.rand((N // 128, K // 128), dtype=torch.bfloat16, device=self.device) * 0.01 + 1e-3
        w_fp32 = self._dequant_weight(w, s, K, N)
        layer = DummyLayer(w, s, K, N)
        state = _build_int8_state(layer, w, s, (128, 128))
        bsc = state["b_scales"].reshape(-1)
        shifts = torch.zeros((K // 128, N // 128), dtype=torch.float32, device=self.device)

        x = torch.randn((M, K), dtype=torch.bfloat16, device=self.device)
        a_i8, a_sc = quant_per_token(x)
        idx = _arange(self.device, N)
        out_sk = lm_head_gemm(a_i8, state["b_col"], a_sc.reshape(-1), bsc, shifts, idx, None, torch.bfloat16)
        out_ref = self._ref_int8(a_i8, a_sc, state["b_col"], bsc)
        assert self._cota_esquema(out_ref, x.float() @ w_fp32.t()) > 0.5

        cos_sim = torch.nn.functional.cosine_similarity(out_sk.float().reshape(-1), out_ref.float().reshape(-1), dim=0).item()
        assert cos_sim >= 0.9995, f"SK-07 cos_sim={cos_sim:.6f} falló para M={M}"

    @pytest.mark.parametrize("M", [1, 4, 8, 16, 32, 64, 69, 128, 512, 1024])
    def test_sk10_mtp_draft_accuracy(self, M):
        """Valida SK-10 (MTP Draft Fused) contra referencia FP32."""
        K, N = 5120, 5120
        torch.manual_seed(42)
        w = torch.randint(-127, 127, (N, K), dtype=torch.int8, device=self.device).to(torch.float8_e4m3fn)
        s = torch.rand((N // 128, K // 128), dtype=torch.bfloat16, device=self.device) * 0.01 + 1e-3
        w_fp32 = self._dequant_weight(w, s, K, N)
        layer = DummyLayer(w, s, K, N)
        state = _build_int8_state(layer, w, s, (128, 128))
        bsc = state["b_scales"].reshape(-1)
        shifts = torch.zeros((K // 128, N // 128), dtype=torch.float32, device=self.device)

        x = torch.randn((M, K), dtype=torch.bfloat16, device=self.device)
        a_i8, a_sc = quant_per_token(x)
        out_sk = mtp_draft_fused_gemm(a_i8, state["b_col"], a_sc.reshape(-1), bsc, shifts, None, torch.bfloat16)
        out_ref = self._ref_int8(a_i8, a_sc, state["b_col"], bsc)
        assert self._cota_esquema(out_ref, x.float() @ w_fp32.t()) > 0.5

        cos_sim = torch.nn.functional.cosine_similarity(out_sk.float().reshape(-1), out_ref.float().reshape(-1), dim=0).item()
        assert cos_sim >= 0.9995, f"SK-10 cos_sim={cos_sim:.6f} falló para M={M}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
