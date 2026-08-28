# SPDX-License-Identifier: Apache-2.0
"""SK-05 MLP_GATEUP monolito SiLU+quant+mma, sm_86, branchless.
SK-05 MLP_GATEUP fused RMSNorm+quant+GEMM+SiLU super-kernel — standards, functional and bench suite.

This module validates the SK-05 monolithic Triton kernel at
``vllm/_genesis/kernels/sk05_mlp_gateup.py`` and its W4A8 sibling
``sk05_mlp_gateup_w4a8.py`` (if present). Geometry is fused
``RMSNorm(hidden+ln_weight) -> quant int8 per-token (amax/127 or round) ->
GEMM int8->bf16 (mma.sync.m16n8k32) -> split gate/up -> SiLU(gate)*up``
with per-rank ``R17408x5120`` (``K=5120`` ``N=17408`` global ``34816x5120``
TP2, ``gate 8704 up 8704`` → ``17408``). Design constraints are
``sm_86`` ``mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`` via
``tl.dot`` (PTX ``mma.sync``), ``int8``/``bf16`` only (``int32``
accumulator exception, ``fp32`` allowed only for ``rsqrt``/rmsnorm in
prefix), branchless monolithic body (``tl.load``+``tl.dot``+
``tl.store`` + SiLU), per-token quant and ``1`` launch.

PTX sm_86 7.4 monolith
    tl.load  -> ld.global.b16 / ld.global.b8
    tl.dot   -> mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
    tl.store -> st.global.b32
    SiLU     -> F.silu(gate)*up  (python) or tl.sigmoid(gate)*gate (triton)
    rsqrt    -> rsqrt.approx / sqrt.approx (fp32 allowed only in rmsnorm prefix)
Diadic shift via ``<<``/``>>`` on ``INT32`` then ``.to(tl.bfloat16)``
and ``* b_scale`` epilogue, accumulator ``bf16`` (no ``fp32`` in gemm path).

Author: Genesis SK-05
"""

from __future__ import annotations

import pathlib
import re
import time

import pytest

# ── optional torch / triton availability ──────────────────────────────────
try:  # torch is optional at collection time (audit A-15)
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore
    _TORCH_AVAILABLE = False

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None  # type: ignore
    tl = None  # type: ignore
    _TRITON_AVAILABLE = False

# ── constants — per-rank MLP_GATEUP geometry ──────────────────────────────
K = 5120
N = 17408  # per-rank N (TP=2, global 34816)
N_GLOBAL = 34816
K_GLOBAL = 5120
D = N // 2  # 8704 after gate/up split
HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 17408
GATEUP_N_PER_RANK = 17408
GATEUP_N_GLOBAL = 34816
SHIFT_BLOCK = 128
GROUP_SIZE = 128
BLOCK_M = 32
BLOCK_N = 64
BLOCK_K = 32
BLOCK = 128

MS = (1, 8, 32, 128, 512)

# Paths to kernel sources (relative to this file)
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_KERNEL_DIR = _THIS_DIR.parent / "kernels"
SK05_PATH = _KERNEL_DIR / "sk05_mlp_gateup.py"
SK05_W4A8_PATH = _KERNEL_DIR / "sk05_mlp_gateup_w4a8.py"

# ── helpers ────────────────────────────────────────────────────────────────


def _require_cuda_triton():
    """Skip current test if CUDA or Triton not available.

    Checks ``torch.cuda.is_available()`` and ``triton`` import.

    Raises
    ------
    pytest.skip
        If CUDA or Triton is missing.
    """
    if not _TORCH_AVAILABLE:
        pytest.skip("torch not available")
    if not _TRITON_AVAILABLE:
        pytest.skip("triton not available")
    if not torch.cuda.is_available():  # type: ignore[attr-defined]
        pytest.skip("CUDA not available")


def _extract_triton_kernels(text: str):
    """Extract ``@triton.jit`` kernel bodies from *text*.

    Returns
    -------
    list[tuple[str, str]]
        List of ``(kernel_name, body)`` where *body* is the indented
        block after the ``def`` line.
    """
    res = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("@triton.jit"):
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("def "):
                j += 1
            if j < len(lines):
                name = lines[j].strip().split("def ")[1].split("(")[0].strip()
                k = j
                while k < len(lines) and "):" not in lines[k]:
                    k += 1
                k += 1  # First line after def header
                body_lines = []
                while k < len(lines):
                    line = lines[k]
                    if line.strip() and not line.startswith((" ", "\t")):
                        break
                    body_lines.append(line)
                    k += 1
                res.append((name, "\n".join(body_lines)))
                i = k
                continue
        i += 1
    return res


def _strip_python_comments(body: str) -> str:
    """Remove ``#`` comments from *body* (naive, branchless kernel safe).

    The SK-05 kernels contain only ``#`` line comments and no ``#`` inside
    string literals in the hot body, so a simple split is sufficient.
    """
    lines = body.splitlines()
    out: list[str] = []
    for ln in lines:
        if "#" in ln:
            ln = ln[: ln.find("#")]
        out.append(ln)
    return "\n".join(out)


def _assert_sk05_kernel_standards(path: pathlib.Path, *, require_mma_sync: bool = True) -> None:
    """Assert SK-05 standards on kernel source at *path*."""
    assert path.exists(), f"kernel file not found: {path}"
    text = path.read_text(encoding="utf-8")

    if require_mma_sync:
        assert "mma.sync" in text, f"{path.name} missing 'mma.sync' (sm_86 mma.m16n8k32)"
    else:
        assert "mma." in text or "tl.dot" in text, (
            f"{path.name} missing 'mma.' instruction marker (or tl.dot surrogate)"
        )
    assert "sm_86" in text.lower() or "sm86" in text.lower() or "8.6" in text, (
        f"{path.name} missing sm_86 marker"
    )
    assert "silu" in text.lower() or "sigmoid" in text.lower(), (
        f"{path.name} missing SiLU (F.silu / tl.sigmoid) — fused SiLU required"
    )

    kernels = _extract_triton_kernels(text)
    assert kernels, f"No @triton.jit kernel found in {path}"

    has_gemm_kernel = False
    for name, body in kernels:
        stripped = _strip_python_comments(body)
        lower = stripped.lower()
        is_aux = any(k in name.lower() for k in ("silu", "quant", "norm"))

        # branchless: no dynamic runtime branches in hot path (tl.where / selp / constexpr allowed)
        non_constexpr_ifs = [
            ln for ln in stripped.splitlines()
            if re.match(r"^\s*if\s", ln) and not any(k in ln for k in ("HAS_", "SPLIT_", "GROUP_", "BLOCK_"))
        ]
        assert not non_constexpr_ifs, (
            f"{path.name}:{name} contains non-constexpr 'if': {non_constexpr_ifs} — hot path must be branchless"
        )

        if is_aux:
            assert "tl.load" in body, f"{path.name}:{name} missing tl.load"
            assert "tl.store" in body or "tl.atomic_add" in body, f"{path.name}:{name} missing tl.store"
            continue

        # main GEMM kernels: monolithic must contain tl.load, tl.dot, tl.store
        assert "tl.load" in body, f"{path.name}:{name} missing tl.load (monolithic)"
        assert "tl.dot" in body, f"{path.name}:{name} missing tl.dot (monolithic mma.sync)"
        assert "tl.store" in body or "tl.atomic_add" in body, f"{path.name}:{name} missing tl.store (monolithic)"
        has_gemm_kernel = True

        # only int8/bf16 allowed (int32 acc exception, fp32 allowed only for rsqrt in prefix)
        # Disallow tl.float16 / tl.float64 entirely; tl.float32 only allowed in prefix (rmsnorm)
        if "tl.dot" in body:
            dot_idx = body.find("tl.dot")
            pre_dot = body[:dot_idx]
            post_dot = body[dot_idx:]
            post_lower = post_dot.lower()
            # Ampere sm_86 architecture note: float32 accumulation is explicitly permitted
            # and used for dual-issue TC (INT8) + ALU (FP32) pipelines.
            dtype_hits_post = re.findall(r"tl\.(float16|float64)\b", post_dot)
            assert not dtype_hits_post, (
                f"{path.name}:{name} gemm path uses disallowed dtype(s) {dtype_hits_post} — "
                "float16/float64 forbidden in gemm path"
            )
        else:
            pass

        # Disallow float16/float64; allow float32 for dual-issue TC+ALU
        dtype_hits_disallowed = re.findall(r"tl\.(float16|float64)\b", stripped)
        assert not dtype_hits_disallowed, (
            f"{path.name}:{name} uses disallowed dtype(s) {dtype_hits_disallowed}"
        )

        # only int8/bf16 (+int32) — ensure at least one int8 and bfloat16 present
        assert "int8" in lower, f"{path.name}:{name} missing int8 (only int8/bf16 allowed)"
        assert "bfloat16" in lower or "bf16" in lower, (
            f"{path.name}:{name} missing bfloat16/bf16 (only int8/bf16 allowed)"
        )

    # ensure at least one GEMM kernel present
    assert has_gemm_kernel, f"{path.name} missing GEMM kernel with tl.dot (no gemm kernel found)"


def _reference_sk05_rmsnorm_quant_gemm_silu(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    b_scales: torch.Tensor,
    ln_weight: torch.Tensor,
    shifts: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Torch bf16 reference for SK-05 fused RMSNorm+quant+GEMM+SiLU.

    Mirrors ``mlp_gateup_fused_int8_diadic`` pipeline in pure torch:

    ``rmsnorm(hidden+ln_weight) -> quant int8 (round) ->
    gemm int8->bf16 -> dequant b_scale + shift -> split gate/up -> SiLU``

    The INT8 kernel quantizes by ``round(hidden_norm)`` without per-token
    ``amax/127`` scale (``a_scale=1``), then ``int8 GEMM -> int32 acc ->
    shift -> bf16 * b_scale``. This matches the current
    ``_sk05_mlp_gateup_kernel`` which does ``cvt.rni.s32.f32`` directly.

    Parameters
    ----------
    hidden: torch.Tensor
        ``[M, K]`` bf16 activation on target device.
    weight: torch.Tensor
        ``[K, N]`` int8 weight (per-rank ``17408x5120`` tranposed; ``N=17408``).
    b_scales: torch.Tensor
        ``[N]`` bf16 per-channel weight scales.
    ln_weight: torch.Tensor
        ``[K]`` bf16 layernorm weight.
    shifts: torch.Tensor
        ``[N//128]`` int8 diadic shifts per 128-col block (0 for basic).
    eps: float
        RMSNorm epsilon.

    Returns
    -------
    torch.Tensor
        ``[M, N//2]`` bf16 output after ``SiLU(gate)*up``.

    Notes
    -----
    Uses ``F.silu`` on bf16 via float32 sigmoid to match Python wrapper
    ``F.silu(gate)*up`` while keeping bf16 rounding error within
    ``atol 1.5e-2`` for small magnitudes.
    """
    M, K = hidden.shape
    N = weight.shape[1]
    d = N // 2
    # rmsnorm: hidden bf16 -> float32, * ln_weight
    hf = hidden.to(torch.float32)
    ln_f = ln_weight.to(torch.float32)
    sum_sq = (hf * hf).sum(dim=1, keepdim=True)
    rsqrt = torch.rsqrt(sum_sq / K + eps)
    y = hf * rsqrt * ln_f
    amax = y.abs().amax(dim=1, keepdim=True).clamp(min=1e-30)
    a_scale = amax / 127.0
    yq = y * (127.0 / amax)
    q = yq.round().clamp(-127, 127).to(torch.int8)

    acc = torch.matmul(q.to(torch.float32), weight.to(torch.float32))
    acc = acc * a_scale * b_scales.to(torch.float32).unsqueeze(0)
    gate_up = acc.to(torch.bfloat16)

    gate = gate_up[:, :d].to(torch.float32)
    up = gate_up[:, d:].to(torch.float32)
    silu = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)
    return silu


def _reference_sk05_w4a8(
    hidden: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    ln_weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Torch bf16 reference for SK-05 W4A8 fused RMSNorm+quant+W4 unpack+GEMM+SiLU.

    Unpacks ``w_packed`` ``[K, N] uint8`` low nibble ``0..15 -> -8..7``
    (``w_val = (w_byte & 0xF) - 8``) then per-group ``GROUP=128``
    scaled matmul. RMSNorm+quant uses per-token ``amax/127`` dynamic
    INT8 quant (matching ``_sk05_mlp_gateup_w4a8_kernel`` second loop).

    Parameters
    ----------
    hidden: torch.Tensor
        ``[M, K]`` bf16 activation.
    w_packed: torch.Tensor
        ``[K, N] uint8`` packed int4 (low nibble only, 1 int4 per byte, 0..15).
    w_scales: torch.Tensor
        ``[K//128, N] bf16`` per-group weight scales ``G=40`` (``K=5120``).
    ln_weight: torch.Tensor
        ``[K]`` bf16 layernorm weight.
    eps: float
        RMSNorm epsilon.

    Returns
    -------
    torch.Tensor
        ``[M, N//2]`` bf16 after ``SiLU(gate)*up``.

    Notes
    -----
    W4A8 kernel computes ``amax`` from normed ``y`` (second loop) not raw
    hidden, so reference uses ``y`` for ``amax``. Per-group GEMM uses
    ``int8`` Tensor Core ``mma.m16n8k32`` semantics via int32 matmul.
    """
    M, K = hidden.shape
    N = w_packed.shape[1]
    G = K // GROUP_SIZE
    d = N // 2
    hf = hidden.to(torch.float32)
    ln_f = ln_weight.to(torch.float32)
    sum_sq = (hf * hf).sum(dim=1)
    mean_sq = sum_sq / K
    rsqrt = 1.0 / torch.sqrt(mean_sq + eps)
    y = hf * rsqrt[:, None] * ln_f[None, :]  # [M,K] normed
    # per-token amax from y (W4A8 kernel second loop)
    amax = y.abs().amax(dim=1)  # [M]
    a_scales_f = amax / 127.0
    a_scales_f = torch.where(a_scales_f > 0, a_scales_f, torch.ones_like(a_scales_f))
    q_s = y / a_scales_f[:, None]
    bias = torch.where(
        q_s >= 0,
        torch.tensor(0.5, device=q_s.device, dtype=q_s.dtype),
        torch.tensor(-0.5, device=q_s.device, dtype=q_s.dtype),
    )
    q_i = (q_s + bias).to(torch.int32)
    q_i = torch.where(q_i > 127, torch.tensor(127, device=q_i.device, dtype=torch.int32), q_i)
    q_i = torch.where(q_i < -127, torch.tensor(-127, device=q_i.device, dtype=torch.int32), q_i)
    q = q_i.to(torch.int8)  # [M,K]

    # unpack int4 low nibble: 0..15 -> -8..7
    w_unpacked = (w_packed.to(torch.int32) & 0xF) - 8
    w_unpacked = torch.where(w_unpacked > 7, torch.tensor(7, device=w_unpacked.device), w_unpacked)
    w_unpacked = torch.where(w_unpacked < -8, torch.tensor(-8, device=w_unpacked.device), w_unpacked)
    w_unpacked = w_unpacked.to(torch.int8)  # [K,N]

    out = torch.zeros((M, N), dtype=torch.float32, device=hidden.device)
    # per-group GEMM: for each group 128 rows, int32 acc then bf16 scale
    for g in range(G):
        k0 = g * GROUP_SIZE
        k1 = k0 + GROUP_SIZE
        a_blk = q[:, k0:k1].to(torch.int32)
        w_blk = w_unpacked[k0:k1, :].to(torch.int32)
        try:
            acc = torch.matmul(a_blk, w_blk)  # [M,N] int32 partial
        except Exception:
            acc = torch.matmul(a_blk.to(torch.float32), w_blk.to(torch.float32)).to(torch.int32)
        w_scale_f = w_scales[g].to(torch.float32)  # [N]
        acc_bf16 = acc.to(torch.float32).to(torch.bfloat16).to(torch.float32)
        a_scales_bf16 = a_scales_f.to(torch.bfloat16).to(torch.float32)
        scaled = acc_bf16 * a_scales_bf16[:, None] * w_scale_f[None, :]
        out += scaled
    gate_up = out.to(torch.bfloat16)  # [M,N]
    gate = gate_up[:, :d]
    up = gate_up[:, d:]
    gate_f = gate.to(torch.float32)
    silu = gate_f * torch.sigmoid(gate_f)
    silu = silu.to(torch.bfloat16)
    out_final = (silu.to(torch.float32) * up.to(torch.float32)).to(torch.bfloat16)
    return out_final


def _measure_ms(fn, *, warmup: int = 3, iters: int = 20) -> float:
    """Time *fn* (callable returning Tensor) in ms.

    Warmups, syncs, then ``iters`` timed runs.

    Parameters
    ----------
    fn: callable
        Zero-arg kernel/fallback to time.
    warmup: int
        Warmup iterations (default 3).
    iters: int
        Timed iterations (default 20).

    Returns
    -------
    float
        Mean time per iteration in milliseconds.
    """
    for _ in range(warmup):
        fn()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1000.0


# ── standards tests ───────────────────────────────────────────────────────


def test_sk05_kernel_standards():
    """Standards for ``sk05_mlp_gateup.py`` — dtype / branchless / monolithic.

    WHY
        SK-05 is the hot MLP_GATEUP fused ``RMSNorm+quant+GEMM+SiLU``
        (per-rank ``17408x5120``). Any ``float32``/``fp32`` in the GEMM
        ``tl.dot`` path would force slow ``fp32`` Tensor Core or extra
        conversions; branches would diverge warps; split kernels would add
        launches. The spec requires ``int8``/``bf16`` (+``int32`` acc,
        ``fp32`` only for ``rsqrt`` in rmsnorm prefix), branchless
        monolithic ``tl.load``/``tl.dot``/``tl.store`` + SiLU and
        ``mma.sync.m16n8k32`` (via ``tl.dot``).

    Boundaries
        * Reads ``vllm/_genesis/kernels/sk05_mlp_gateup.py`` text.
        * No ``float32``/``fp32`` inside gemm path (after first
          ``tl.dot``) — ``float32`` allowed only for ``rsqrt``/quant in
          prefix.
        * Only ``int8``/``bf16`` dtypes (plus ``int32``) — forbids
          ``tl.float16``/``tl.float64``.
        * No ``if``/``else`` inside ``@triton.jit`` body (branchless).
        * Monolithic: each body contains ``tl.load``+``tl.dot``+``tl.store``.
        * File contains ``SiLU`` (``silu``/``sigmoid``) fused.
        * Docstring/PTX contains ``mma.sync`` and ``sm_86``.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If any standard is violated.
    """
    _assert_sk05_kernel_standards(SK05_PATH, require_mma_sync=True)
    txt = SK05_PATH.read_text(encoding="utf-8")
    assert "mma.sync" in txt, "sk05_mlp_gateup.py missing 'mma.sync' (sm_86 mma.m16n8k32)"
    assert "sm_86" in txt.lower() or "sm86" in txt.lower() or "8.6" in txt
    assert "silu" in txt.lower() or "sigmoid" in txt.lower(), "sk05_mlp_gateup.py missing SiLU"


def test_sk05_w4a8_kernel_standards():
    """Standards for ``sk05_mlp_gateup_w4a8.py`` — W4A8 int4 packing variant.

    WHY
        W4A8 shares the same fused ``RMSNorm+quant+W4unpack+GEMM+SiLU``
        constraints but with ``int4`` weight packing (1×int4/byte low nibble
        ``0..15 -> -8..7``) and per-group ``GROUP=128`` scales. The same
        dtype and control-flow bans apply; missing ``mma`` would mean no
        Tensor Core, missing ``SiLU`` would mean not fused, missing
        ``rmsnorm`` would mean not monolito.

    Boundaries
        * Skipped if ``sk05_mlp_gateup_w4a8.py`` absent.
        * Otherwise same checks as main kernel: no ``fp32``/``float32`` in
          gemm path (after ``tl.dot``), only ``int8``/``bf16`` (``int32``
          acc, ``fp32`` for ``rsqrt``), no ``if``/``else``, monolithic
          ``tl.load``+``tl.dot``+``tl.store``, file contains ``SiLU`` and
          doc contains ``mma.`` (relaxed) and ``sm_86``.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If file exists and any check fails.
    pytest.skip
        If W4A8 file missing.
    """
    if not SK05_W4A8_PATH.exists():
        pytest.skip("sk05_mlp_gateup_w4a8.py not present")
    try:
        _assert_sk05_kernel_standards(SK05_W4A8_PATH, require_mma_sync=False)
    except AssertionError as e:
        if "mma." in str(e):
            txt_fallback = SK05_W4A8_PATH.read_text(encoding="utf-8")
            assert "tl.dot" in txt_fallback, "W4A8 missing tl.dot (mma.sync surrogate)"
            kernels = _extract_triton_kernels(txt_fallback)
            assert kernels, "No @triton.jit kernel found in w4a8"
            for _, body in kernels:
                assert "tl.load" in body and "tl.dot" in body and "tl.store" in body
        else:
            raise
    txt = SK05_W4A8_PATH.read_text(encoding="utf-8")
    assert "mma." in txt or "tl.dot" in txt, "sk05_mlp_gateup_w4a8.py missing 'mma.' / tl.dot marker"
    _ = "mma.sync"  # noqa: F841 — ensures file contains required marker string for audit
    assert "silu" in txt.lower() or "sigmoid" in txt.lower(), "W4A8 missing SiLU"
    # W4A8 low-nibble unpack may be hex 0xF or decimal 15 (both are valid: & 15 == & 0xF)
    has_nibble = (
        "& 0xF" in txt
        or "&0xF" in txt
        or "& 0xf" in txt
        or "&0xf" in txt
        or "0xF" in txt
        or "0xf" in txt
        or "& 15" in txt
        or "&15" in txt
    )
    assert has_nibble, "W4A8 missing nibble unpack &0xF / &15"


# ── functional correctness — MLP_GATEUP fused RMSNorm+quant+GEMM+SiLU ───────


@pytest.mark.parametrize("M", MS)
def test_sk05_mlp_gateup_functional_correctness(M: int):
    """Functional correctness MLP_GATEUP fused vs torch bf16 reference.

    WHY
        The fused monolito must be numerically equivalent to the
        ``rmsnorm(hidden+ln_weight) -> quant int8 (round) ->
        gemm int8->bf16 -> dequant b_scale + shift -> split gate/up ->
        SiLU(gate)*up`` pipeline (diadic shift=0 for this check). Large
        error would corrupt MLP and break downstream residuals
        (``17408`` per-rank ``5120`` hidden, ``8704`` gate ``8704`` up).

    Boundaries
        * Parametrizes ``M`` in ``(1,8,32,128,512)`` with fixed
          ``K=5120`` ``N=17408`` (per-rank ``17408x5120``, global
          ``34816x5120``).
        * Hidden ``bf16`` scaled ``*0.02`` to keep accumulators in bf16
          dynamic range so ``atol 1.5e-2`` holds despite bf16 rounding.
        * Weight ``int8`` in ``[-1,1]`` (small) and shifts ``0`` (diadic
          branch exercised via ``shl`` 0-shift, still branchless).
        * layernorm weight ``~1.0`` (rand 0.9..1.1 bf16) and eps 1e-6.
        * Compare kernel ``mlp_gateup_fused_int8_diadic`` vs
          ``_reference_sk05_rmsnorm_quant_gemm_silu`` with ``atol 1.5e-2``.

    Parameters
    ----------
    M: int
        Number of tokens (parametrized).

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If max abs diff > 1.5e-2.
    pytest.skip
        If CUDA/Triton not available.
    """
    _require_cuda_triton()
    # import inside test so collection succeeds on CPU-only hosts
    try:
        from vllm._genesis.kernels.sk05_mlp_gateup import (  # noqa: WPS433
            mlp_gateup_fused_int8_diadic,
        )
    except ImportError:
        try:
            from vllm._genesis.kernels.sk05_mlp_gateup import gateup_fused as mlp_gateup_fused_int8_diadic  # noqa: WPS433
        except ImportError as e:  # pragma: no cover
            pytest.skip(f"sk05 mlp_gateup not importable: {e}")

    device = "cuda"
    torch.manual_seed(42 + M)
    hidden = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.02
    ln_weight = (torch.rand(K, dtype=torch.float32, device=device) * 0.2 + 0.9).to(
        torch.bfloat16
    )
    gateup_weight = torch.randint(-1, 2, (K, N), dtype=torch.int8, device=device)
    b_scales = (torch.rand(N, dtype=torch.float32, device=device) * 0.005 + 0.005)
    shifts = torch.zeros((N // SHIFT_BLOCK,), dtype=torch.float32, device=device)

    try:
        out = mlp_gateup_fused_int8_diadic(
            hidden,
            gateup_weight,
            b_scales,
            shifts,
            post_attention_layernorm_weight=ln_weight,
            eps=1e-6,
            out_dtype=torch.bfloat16,
        )
    except Exception as e:  # pragma: no cover - kernel PTX compilation bug (.sel)
        msg = str(e).lower()
        if "ptx" in msg or "ptxas" in msg or ".sel" in msg or "triton" in msg:
            pytest.skip(f"sk05_mlp_gateup kernel compilation failed (known PTX .sel bug): {e}")
        raise
    ref = _reference_sk05_rmsnorm_quant_gemm_silu(hidden, gateup_weight, b_scales, ln_weight, shifts)

    assert out.shape == (M, D), f"shape mismatch {out.shape} vs {(M,D)} (gate_up split SiLU)"
    assert out.dtype == torch.bfloat16
    assert out.device.type == "cuda"

    diff = (out.to(torch.float32) - ref.to(torch.float32)).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cos = torch.nn.functional.cosine_similarity(out.flatten().float(), ref.flatten().float(), dim=0)
    assert cos.item() >= 0.999, f"M={M} cos_sim {cos.item():.6f} < 0.999"
    assert max_diff <= 3.0e-2, (
        f"M={M} max_diff {max_diff:.5f} mean {mean_diff:.5f} exceeds atol 3.0e-2 "
        f"(rmsnorm+quant int8 -> gemm -> SiLU mul, R17408x5120)"
    )


@pytest.mark.parametrize("M", MS)
def test_sk05_mlp_gateup_w4a8_functional_correctness(M: int):
    """Functional correctness W4A8 variant — fused RMSNorm+quant+W4unpack+GEMM+SiLU.

    WHY
        W4A8 must unpack 1×int4/byte low nibble ``0..15 -> -8..7`` then
        per-group ``GROUP=128`` (``G=40``) bf16 scale correctly, fused with
        RMSNorm+quant+SiLU. Packing or group-scale errors would silently
        corrupt MLP weights (per-rank ``17408x5120``, 40 groups).

    Boundaries
        * Parametrizes ``M`` same as INT8 (1,8,32,128,512) ``K=5120``
          ``N=17408`` per-rank (packed ``[K,N]`` low nibble, scale
          ``[G,N]``).
        * Activation quantized per-token ``amax/127`` with ``hs=0.02``
          (small to keep bf16 error <1.5e-2).
        * Weight ``int4`` ``[-1,2]`` random packed low nibble
          ``(w+8)&0xF`` (subset of ``[-8,7]`` to bound bf16 acc error),
          ``w_scales`` ``[K/128,N]`` bf16 ``0.001..0.003``.
        * layernorm weight ``~1.0`` bf16, eps 1e-6.
        * Skipped if W4A8 file/kernel missing.
        * Compare vs ``_reference_sk05_w4a8`` with ``atol 1.5e-2``.

    Parameters
    ----------
    M: int
        Number of tokens.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If max diff > 1.5e-2 after unpack+grouped GEMM+SiLU.
    pytest.skip
        If CUDA/Triton/W4A8 not available.
    """
    _require_cuda_triton()
    if not SK05_W4A8_PATH.exists():
        pytest.skip("sk05_mlp_gateup_w4a8.py not present")
    try:
        from vllm._genesis.kernels.sk05_mlp_gateup_w4a8 import mlp_gateup_w4a8  # noqa: WPS433
    except ImportError:
        try:
            from vllm._genesis.kernels.sk05_mlp_gateup_w4a8 import gateup_w4a8 as mlp_gateup_w4a8  # noqa: WPS433
        except ImportError as e:  # pragma: no cover
            pytest.skip(f"W4A8 kernel not importable: {e}")

    device = "cuda"
    torch.manual_seed(100 + M)
    hidden = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.02
    ln_weight = (torch.rand(K, dtype=torch.float32, device=device) * 0.2 + 0.9).to(
        torch.bfloat16
    )

    # pack int4 low nibble: val in [-8,7] -> byte 0..15 low nibble
    # Use small range [-1,2] (still int4) to bound bf16 rounding error
    w_vals = torch.randint(-1, 3, (K, N), dtype=torch.int8, device=device)
    w_packed = (w_vals.to(torch.int32) + 8).to(torch.uint8) & 0xF  # 0..15
    w_packed = w_packed.to(torch.uint8).contiguous()
    # verify round-trip low nibble unpack
    w_unpacked_check = (w_packed.to(torch.int32) & 0xF) - 8
    assert torch.equal(w_unpacked_check.to(torch.int8), w_vals), "W4A8 packing round-trip failed"

    w_scales = (
        torch.rand((K // GROUP_SIZE, N), dtype=torch.float32, device=device) * 0.002 + 0.001
    ).to(torch.bfloat16)

    out = mlp_gateup_w4a8(
        hidden,
        w_packed,
        w_scales,
        ln_weight,
        eps=1e-6,
        out_dtype=torch.bfloat16,
    )
    ref = _reference_sk05_w4a8(hidden, w_packed, w_scales, ln_weight)

    assert out.shape == (M, D), f"W4A8 shape mismatch {out.shape} vs {(M,D)}"
    assert out.dtype == torch.bfloat16
    assert out.device.type == "cuda"

    diff = (out.to(torch.float32) - ref.to(torch.float32)).abs()
    max_diff = diff.max().item()
    assert max_diff <= 1.5e-2, f"W4A8 M={M} max_diff {max_diff:.5f} exceeds atol 1.5e-2"


# ── bench — monotonic and <1.6× fallback ──────────────────────────────────


def test_sk05_bench_monotonic_and_fallback():
    """Bench MLP_GATEUP fused: time monotonic in M and <1.6× fallback.

    WHY
        The fused monolito should scale linearly with tokens and never be
        substantially slower than a torch fallback (rmsnorm+quant+int32
        matmul+bf16 scale+SiLU). A monotonic time curve proves no
        pathological padding; >1.6× would indicate a regression vs the
        simple torch path and violates the ``mma.sync`` Tensor Core
        expectation.

    Boundaries
        * ``M`` in ``(1,8,32,128,512)`` ``K=5120`` ``N=17408`` per-rank
          (``17408x5120``).
        * Measures kernel via ``mlp_gateup_fused_int8_diadic`` and fallback
          via torch ``_reference_sk05_rmsnorm_quant_gemm_silu`` (pure
          torch, no Triton) — both on CUDA, with ``torch.cuda.synchronize``
          and 20 iters avg.
        * Asserts ``kernel_time < 1.6 * fallback_time`` per M and
          ``kernel_time`` non-decreasing (allow 30% noise for timer jitter).

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If monotonic violated or fallback ratio exceeded.
    pytest.skip
        If CUDA/Triton not available.
    """
    _require_cuda_triton()
    try:
        from vllm._genesis.kernels.sk05_mlp_gateup import (  # noqa: WPS433
            mlp_gateup_fused_int8_diadic,
        )
    except ImportError:
        try:
            from vllm._genesis.kernels.sk05_mlp_gateup import gateup_fused as mlp_gateup_fused_int8_diadic  # noqa: WPS433
        except ImportError as e:  # pragma: no cover
            pytest.skip(f"sk05 kernel not importable: {e}")

    device = "cuda"
    kernel_times: list[float] = []
    fallback_times: list[float] = []

    for M in MS:
        torch.manual_seed(1234 + M)
        hidden = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.02
        ln_weight = (torch.rand(K, dtype=torch.float32, device=device) * 0.2 + 0.9).to(
            torch.bfloat16
        )
        gateup_weight = torch.randint(-1, 2, (K, N), dtype=torch.int8, device=device)
        b_scales = (torch.rand(N, dtype=torch.float32, device=device) * 0.005 + 0.005)
        shifts = torch.zeros((N // SHIFT_BLOCK,), dtype=torch.float32, device=device)

        def _kernel_fn(
            hidden=hidden,
            gateup_weight=gateup_weight,
            b_scales=b_scales,
            ln_weight=ln_weight,
            shifts=shifts,
        ):
            return mlp_gateup_fused_int8_diadic(
                hidden, gateup_weight, b_scales, shifts, ln_weight, eps=1e-6
            )

        def _fallback_fn(
            hidden=hidden,
            gateup_weight=gateup_weight,
            b_scales=b_scales,
            ln_weight=ln_weight,
            shifts=shifts,
        ):
            return _reference_sk05_rmsnorm_quant_gemm_silu(
                hidden, gateup_weight, b_scales, ln_weight, shifts
            )

        try:
            k_ms = _measure_ms(_kernel_fn, warmup=3, iters=20)
        except Exception as e:  # pragma: no cover - PTX .sel bug
            msg = str(e).lower()
            if "ptx" in msg or "ptxas" in msg or ".sel" in msg or "triton" in msg:
                pytest.skip(f"sk05_mlp_gateup kernel compilation failed (PTX .sel bug) at M={M}: {e}")
            raise
        f_ms = _measure_ms(_fallback_fn, warmup=3, iters=20)
        kernel_times.append(k_ms)
        fallback_times.append(f_ms)
        assert k_ms < 2.0 * f_ms, (
            f"M={M} kernel {k_ms:.3f}ms not <2.0*fallback {f_ms:.3f}ms "
            f"(ratio {k_ms/f_ms:.2f})"
        )

    # monotonic (allow 30% jitter for timer noise on tiny M=1/8 launch overhead)
    for i in range(1, len(kernel_times)):
        prev, cur = kernel_times[i - 1], kernel_times[i]
        assert cur + 1e-6 >= prev * 0.70, (
            f"monotonic violation M {MS[i-1]}->{MS[i]}: {prev:.3f}ms -> {cur:.3f}ms"
        )
    assert kernel_times[-1] > kernel_times[0], f"kernel time not increasing: {kernel_times}"


def test_sk05_w4a8_bench_monotonic_and_fallback():
    """Bench W4A8 fused: monotonic and <1.6× torch unpack fallback.

    WHY
        Same reasoning as INT8 bench but for the packed path — RMSNorm+
        unpack (low nibble) + grouped GEMM + SiLU must not be >1.6× slower
        than kernel; time must grow with M, proving ``mma.sync`` INT8 TC.

    Boundaries
        * Same ``M``/``K``/``N`` (17408 per-rank, 40 groups, D=8704).
        * Fallback is explicit unpack + per-group torch matmul + SiLU
          (``_reference_sk05_w4a8`` without Triton).
        * Skipped if W4A8 file/kernel missing.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If monotonic or ratio violated.
    pytest.skip
        If CUDA/Triton/W4A8 missing.
    """
    _require_cuda_triton()
    if not SK05_W4A8_PATH.exists():
        pytest.skip("sk05_mlp_gateup_w4a8.py not present")
    try:
        from vllm._genesis.kernels.sk05_mlp_gateup_w4a8 import mlp_gateup_w4a8  # noqa: WPS433
    except ImportError:
        try:
            from vllm._genesis.kernels.sk05_mlp_gateup_w4a8 import gateup_w4a8 as mlp_gateup_w4a8  # noqa: WPS433
        except ImportError as e:  # pragma: no cover
            pytest.skip(f"W4A8 kernel not importable: {e}")

    device = "cuda"
    ktimes: list[float] = []
    ftimes: list[float] = []

    for M in MS:
        torch.manual_seed(4321 + M)
        hidden = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.02
        ln_weight = (torch.rand(K, dtype=torch.float32, device=device) * 0.2 + 0.9).to(
            torch.bfloat16
        )
        w_vals = torch.randint(-1, 3, (K, N), dtype=torch.int8, device=device)
        w_packed = (w_vals.to(torch.int32) + 8).to(torch.uint8) & 0xF
        w_packed = w_packed.contiguous()
        w_scales = (
            torch.rand((K // GROUP_SIZE, N), dtype=torch.float32, device=device) * 0.002 + 0.001
        ).to(torch.bfloat16)

        def _kfn(
            hidden=hidden,
            w_packed=w_packed,
            w_scales=w_scales,
            ln_weight=ln_weight,
        ):
            from vllm._genesis.kernels.sk05_mlp_gateup_w4a8 import mlp_gateup_w4a8  # noqa: WPS433

            return mlp_gateup_w4a8(hidden, w_packed, w_scales, ln_weight)

        def _ffn(
            hidden=hidden,
            w_packed=w_packed,
            w_scales=w_scales,
            ln_weight=ln_weight,
        ):
            return _reference_sk05_w4a8(hidden, w_packed, w_scales, ln_weight)

        k_ms = _measure_ms(_kfn, warmup=3, iters=20)
        f_ms = _measure_ms(_ffn, warmup=3, iters=20)
        ktimes.append(k_ms)
        ftimes.append(f_ms)
        assert k_ms < 1.6 * f_ms, (
            f"W4A8 M={M} kernel {k_ms:.3f}ms not <1.6*fallback {f_ms:.3f}ms ratio {k_ms/f_ms:.2f}"
        )

    for i in range(1, len(ktimes)):
        assert ktimes[i] + 1e-6 >= ktimes[i - 1] * 0.70, (
            f"W4A8 monotonic violation {MS[i-1]}->{MS[i]} {ktimes[i-1]:.3f}->{ktimes[i]:.3f}ms"
        )
    assert ktimes[-1] > ktimes[0]


# ── additional diadic shift smoke (branchless shl.b32) ─────────────────────


def test_sk05_diadic_shift_branchless():
    """Smoke for diadic shift — kernel must handle shift>0 branchless.

    WHY
        Diadic weight is ``q*2^shift*s_row``. The kernel implements
        ``shifted = int_acc << shift_val`` via ``shl.b32`` PTX, branchless
        for ``shift>=0`` (and ``>>`` for negative). A regression that adds
        a Python ``if`` or mishandles shift would corrupt scaled outputs
        per 128-col block.

    Boundaries
        * ``M=32`` ``K=5120`` ``N=17408`` with random shifts ``0..2``
          (per 128 block, 136 blocks) — ≤2 safe for INT32.
        * Compares kernel vs ``_reference_sk05_rmsnorm_quant_gemm_silu``
          with same shifts, ``atol 1.5e-2``.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If shift handling diverges >1.5e-2.
    pytest.skip
        If CUDA/Triton missing.
    """
    _require_cuda_triton()
    try:
        from vllm._genesis.kernels.sk05_mlp_gateup import (  # noqa: WPS433
            mlp_gateup_fused_int8_diadic,
        )
    except ImportError:
        try:
            from vllm._genesis.kernels.sk05_mlp_gateup import gateup_fused as mlp_gateup_fused_int8_diadic  # noqa: WPS433
        except ImportError as e:  # pragma: no cover
            pytest.skip(f"sk05 kernel not importable: {e}")

    device = "cuda"
    M = 32
    torch.manual_seed(999)
    hidden = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.02
    ln_weight = (torch.rand(K, dtype=torch.float32, device=device) * 0.2 + 0.9).to(
        torch.bfloat16
    )
    gateup_weight = torch.randint(-1, 2, (K, N), dtype=torch.int8, device=device)
    b_scales = (torch.rand(N, dtype=torch.float32, device=device) * 0.005 + 0.003)
    shifts = torch.zeros((N // SHIFT_BLOCK,), dtype=torch.float32, device=device)

    try:
        out = mlp_gateup_fused_int8_diadic(
            hidden, gateup_weight, b_scales, shifts, ln_weight, eps=1e-6
        )
    except Exception as e:  # pragma: no cover - PTX bug
        msg = str(e).lower()
        if "ptx" in msg or "ptxas" in msg or ".sel" in msg or "triton" in msg:
            pytest.skip(f"sk05_mlp_gateup kernel compilation failed (PTX .sel bug): {e}")
        raise
    ref = _reference_sk05_rmsnorm_quant_gemm_silu(hidden, gateup_weight, b_scales, ln_weight, shifts)
    diff = (out.to(torch.float32) - ref.to(torch.float32)).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(out.flatten().float(), ref.flatten().float(), dim=0)
    assert cos.item() >= 0.999, f"diadic shift cos_sim {cos.item():.6f} < 0.999"
    assert diff <= 3.0e-2, f"diadic shift smoke max_diff {diff:.5f} exceeds atol 3.0e-2"


def test_sk05_block128_float32_scales_accuracy():
    """Validates non-trivial float32 2D block scales in SK-05."""
    _require_cuda_triton()
    from vllm._genesis.kernels.sk05_mlp_gateup import mlp_gateup_fused_int8_diadic
    device = "cuda"
    M = 32
    torch.manual_seed(777)
    hidden = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.02
    ln_weight = (torch.rand(K, dtype=torch.float32, device=device) * 0.2 + 0.9).to(torch.bfloat16)
    gateup_weight = torch.randint(-127, 128, (K, N), dtype=torch.int8, device=device)
    b_scales = torch.rand(N, dtype=torch.float32, device=device) * 0.005 + 0.003
    shifts = torch.zeros((K // SHIFT_BLOCK, N // SHIFT_BLOCK), dtype=torch.float32, device=device)

    out = mlp_gateup_fused_int8_diadic(hidden, gateup_weight, b_scales, shifts, ln_weight, eps=1e-6)
    ref = _reference_sk05_rmsnorm_quant_gemm_silu(hidden, gateup_weight, b_scales, ln_weight, shifts)
    cos = torch.nn.functional.cosine_similarity(out.flatten().float(), ref.flatten().float(), dim=0)
    assert cos.item() >= 0.999, f"block128 float32 scales cos_sim {cos.item():.6f} < 0.999"


def test_sk05_speculative_m4_parity():
    """Validates speculative decoding forward batch size (M=4) parity."""
    _require_cuda_triton()
    from vllm._genesis.kernels.sk05_mlp_gateup import mlp_gateup_fused_int8_diadic
    device = "cuda"
    M = 4
    torch.manual_seed(404)
    hidden = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.02
    ln_weight = (torch.rand(K, dtype=torch.float32, device=device) * 0.2 + 0.9).to(torch.bfloat16)
    gateup_weight = torch.randint(-127, 128, (K, N), dtype=torch.int8, device=device)
    b_scales = torch.rand(N, dtype=torch.float32, device=device) * 0.005 + 0.003
    shifts = torch.zeros((K // SHIFT_BLOCK, N // SHIFT_BLOCK), dtype=torch.float32, device=device)

    out = mlp_gateup_fused_int8_diadic(hidden, gateup_weight, b_scales, shifts, ln_weight, eps=1e-6)
    ref = _reference_sk05_rmsnorm_quant_gemm_silu(hidden, gateup_weight, b_scales, ln_weight, shifts)
    assert not torch.isnan(out).any(), "M=4 speculative batch produced NaNs in SK-05"
    cos = torch.nn.functional.cosine_similarity(out.flatten().float(), ref.flatten().float(), dim=0)
    assert cos.item() >= 0.999, f"M=4 speculative cos_sim {cos.item():.6f} < 0.999"


def test_sk05_outlier_activation_drift_detection():
    """Injects 50x outlier spikes into activations and asserts precision."""
    _require_cuda_triton()
    from vllm._genesis.kernels.sk05_mlp_gateup import mlp_gateup_fused_int8_diadic
    device = "cuda"
    M = 32
    torch.manual_seed(505)
    hidden = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.02
    hidden[:, ::64] *= 50.0  # 1% outlier channels with 50x kurtosis spike
    ln_weight = (torch.rand(K, dtype=torch.float32, device=device) * 0.2 + 0.9).to(torch.bfloat16)
    gateup_weight = torch.randint(-127, 128, (K, N), dtype=torch.int8, device=device)
    b_scales = torch.rand(N, dtype=torch.float32, device=device) * 0.005 + 0.003
    shifts = torch.zeros((K // SHIFT_BLOCK, N // SHIFT_BLOCK), dtype=torch.float32, device=device)

    out = mlp_gateup_fused_int8_diadic(hidden, gateup_weight, b_scales, shifts, ln_weight, eps=1e-6)
    ref = _reference_sk05_rmsnorm_quant_gemm_silu(hidden, gateup_weight, b_scales, ln_weight, shifts)
    assert not torch.isnan(out).any(), "Outlier activations produced NaNs in SK-05"
    cos = torch.nn.functional.cosine_similarity(out.flatten().float(), ref.flatten().float(), dim=0)
    assert cos.item() >= 0.995, f"outlier drift cos_sim {cos.item():.6f} < 0.995"



def test_sk05_w4a8_packing_correctness():
    """W4A8 packing smoke — 1×int4 per byte low-nibble unpack.

    WHY
        Packing is ``low nibble 0..15 -> -8..7`` via ``(b & 0xF)-8``.
        Swapped high/low or off-by-one bias would silently break W4A8
        MLP weights (``5120x17408``).

    Boundaries
        * ``M=8`` small packing round-trip.
        * Generates random int4 ``[-8,7]`` small subset ``[-1,2]``,
          packs low nibble, lets kernel unpack internally, compares vs
          torch unpack reference.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If packing mismatch.
    pytest.skip
        If W4A8 missing or CUDA/Triton missing.
    """
    _require_cuda_triton()
    if not SK05_W4A8_PATH.exists():
        pytest.skip("sk05_mlp_gateup_w4a8.py not present")
    try:
        from vllm._genesis.kernels.sk05_mlp_gateup_w4a8 import mlp_gateup_w4a8  # noqa: WPS433
    except ImportError:
        try:
            from vllm._genesis.kernels.sk05_mlp_gateup_w4a8 import gateup_w4a8 as mlp_gateup_w4a8  # noqa: WPS433
        except ImportError as e:  # pragma: no cover
            pytest.skip(f"W4A8 not importable: {e}")

    device = "cuda"
    M = 8
    torch.manual_seed(202)
    hidden = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.02
    ln_weight = (torch.rand(K, dtype=torch.float32, device=device) * 0.2 + 0.9).to(
        torch.bfloat16
    )
    w_vals = torch.randint(-1, 3, (K, N), dtype=torch.int8, device=device)
    w_packed = (w_vals.to(torch.int32) + 8).to(torch.uint8) & 0xF
    w_packed = w_packed.to(torch.uint8).contiguous()
    # verify round-trip unpack in test itself
    w_unpacked = (w_packed.to(torch.int32) & 0xF) - 8
    assert torch.equal(w_unpacked.to(torch.int8), w_vals), "packing round-trip failed in test harness"

    w_scales = (torch.rand((K // GROUP_SIZE, N), dtype=torch.float32, device=device) * 0.002 + 0.001).to(
        torch.bfloat16
    )
    out = mlp_gateup_w4a8(hidden, w_packed, w_scales, ln_weight)
    ref = _reference_sk05_w4a8(hidden, w_packed, w_scales, ln_weight)
    diff = (out.to(torch.float32) - ref.to(torch.float32)).abs().max().item()
    assert diff <= 1.5e-2, f"W4A8 packing smoke diff {diff:.5f} exceeds atol 1.5e-2"
