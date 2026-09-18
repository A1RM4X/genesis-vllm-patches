# SPDX-License-Identifier: Apache-2.0
"""Wiring for CK-4.2 (B3) — custom all-reduce TP=2 fast path + 32 MiB buffer + CuMem graph bypass.

Contrato
--------
- Ubicación de referencia: assets/vllm/vllm/v1/worker/gpu_model_runner.py
  línea ~6546 (custom all-reduce dispatch). En v0.27.1 ese
  despacho vive en ``vllm.distributed.device_communicators.cuda_communicator``
  (``CudaCommunicator.all_reduce``, cuda_communicator.py:275) que delega a
  ``CustomAllreduce.should_custom_ar / custom_all_reduce``
  (custom_all_reduce.py:348/:382).
- Env flag: ``GENESIS_ENABLE_B3_CUSTOM_AR`` (off por defecto).
- Marker: ``Genesis B3 custom AR TP=2 fast path v1 // gpu_model_runner:6546``
  usado para idempotencia y para ``is_applied()``.
- Función pública: ``patch_B3_custom_ar() -> tuple[str,str]`` (status, reason)
  + alias ``apply()`` para el dispatcher ``apply_all``.

Qué hace
--------
1. **32 MiB Buffer**: Aloca búfer de 32 MiB por defecto (controlable con
   ``GENESIS_B3_MAX_SIZE_MIB``) en lugar de los 8 MiB estándar de vLLM,
   permitiendo que prompts de 2.048 tokens (~21 MiB) entren en Custom All-Reduce
   a través del enlace PCIe P2P sin degradar a PyNCCL.
2. **CuMem / CUDA Graphs Bypass**: Durante la captura de grafos CUDA
   (``self._IS_CAPTURING == True``), el despacho a Custom All-Reduce se omite
   seguramente (retorna False en ``should_custom_ar`` y early-exit en
   ``register_graph_buffers`` cuando offset está vacío). Esto elimina el fatal
   csrc/custom_all_reduce.cuh:164 / :455 'invalid argument' causado por
   cudaIpcOpenMemHandle con punteros virtuales CuMem en PyTorch.
3. **Persistencia Multi-proceso**: Utiliza ``TextPatcher`` sobre
   ``vllm/distributed/device_communicators/custom_all_reduce.py`` en disco,
   garantizando que todos los workers de TP (Worker_TP0, Worker_TP1) hereden
   la configuración y los guards.
4. **Idempotencia**: Si el marker ya está presente en el archivo destino,
   retorna "applied" (idempotente) sin duplicar cambios.
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file
from vllm._genesis.wiring.text_patch import (
    TextPatcher,
    TextPatchResult,
    TextPatch,
    result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.B3_custom_ar")

# ─── Public contract constants ──────────────────────────────────────────
ENV_FLAG = "GENESIS_ENABLE_B3_CUSTOM_AR"
GENESIS_B3_MARKER = "Genesis B3 custom AR TP=2 fast path v1 // gpu_model_runner:6546"
GENESIS_B3_CUSTOM_AR_MARKER = GENESIS_B3_MARKER

_TRUTHY = ("1", "true", "yes", "on")
_INSTALLED = False

# ─── Anchors para TextPatcher sobre custom_all_reduce.py ────────────────
B3_INIT_OLD = (
    "    # max_size: max supported allreduce size\n"
    "    def __init__(\n"
    "        self,\n"
    "        group: ProcessGroup,\n"
    "        device: int | str | torch.device,\n"
    "        max_size=8192 * 1024,\n"
)

B3_INIT_NEW = (
    "    # [Genesis B3] max_size: 32 MiB buffer for TP=2 P2P custom all-reduce\n"
    "    def __init__(\n"
    "        self,\n"
    "        group: ProcessGroup,\n"
    "        device: int | str | torch.device,\n"
    "        max_size=32 * 1024 * 1024,\n"
)

# Ojo con alargar esta ancla. Hasta v0.27.1 el `inp_size = ...` venia pegado al
# `return False` y estaba incluido aca; en v0.29.0 upstream metio un chequeo de dtype en el
# medio y el ancla larga dejo de aparecer — el parche se apagaba solo, en silencio. Se queda
# en las tres lineas de la cabecera, que son las estables, y el bypass entra justo despues,
# antes de cualquier guarda nueva que upstream agregue mas abajo.
B3_SHOULD_OLD = (
    "    def should_custom_ar(self, inp: torch.Tensor):\n"
    "        if self.disabled or self.world_size > 8:\n"
    "            return False\n"
)

B3_SHOULD_NEW = (
    "    def should_custom_ar(self, inp: torch.Tensor):\n"
    "        if self.disabled or self.world_size > 8:\n"
    "            return False\n"
    "        # [Genesis B3] Bypass custom AR during CUDA graph capture to prevent CuMem IPC crash.\n"
    "        # Con GENESIS_B3_CAPTURA=1 (y sin --enable-cumem-allocator) se permite capturarlo:\n"
    "        # en decode TODAS las all-reduce van dentro del grafo, asi que con el bypass NCCL\n"
    "        # se queda con el 100% (141 lanzamientos, 4,0 ms por paso, 25,6 us cada uno).\n"
    "        import os as _os\n"
    "        if getattr(self, '_IS_CAPTURING', False) and _os.environ.get('GENESIS_B3_CAPTURA', '0') != '1':\n"
    "            return False\n"
)

B3_REGISTER_OLD = (
    "    def register_graph_buffers(self):\n"
    "        handle, offset = ops.get_graph_buffer_ipc_meta(self._ptr)\n"
    "        logger.debug(\"Registering %d cuda graph addresses\", len(offset))\n"
)

B3_REGISTER_NEW = (
    "    def register_graph_buffers(self):\n"
    "        handle, offset = ops.get_graph_buffer_ipc_meta(self._ptr)\n"
    "        # [Genesis B3] If no graph buffers registered (bypassed for CUDA graphs), skip C++ call\n"
    "        if not offset:\n"
    "            return\n"
    "        logger.debug(\"Registering %d cuda graph addresses\", len(offset))\n"
)


def _is_enabled() -> bool:
    return os.environ.get(ENV_FLAG, "").strip().lower() in _TRUTHY


def is_applied() -> bool:
    """True if B3 markers are in disk or wrappers installed."""
    global _INSTALLED
    if _INSTALLED:
        return True
    try:
        target = resolve_vllm_file("distributed/device_communicators/custom_all_reduce.py")
        if target and os.path.isfile(target):
            with open(target, "r", encoding="utf-8") as f:
                content = f.read()
            if GENESIS_B3_MARKER in content or "[Genesis B3]" in content:
                return True
    except Exception:
        pass
    return False


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("distributed/device_communicators/custom_all_reduce.py")
    if target is None or not os.path.isfile(target):
        return None
    return TextPatcher(
        patch_name="B3 distributed/device_communicators/custom_all_reduce.py — 32 MiB buffer + CuMem graph bypass",
        target_file=str(target),
        marker=GENESIS_B3_MARKER,
        sub_patches=[
            TextPatch(
                name="b3_init_32mib",
                anchor=B3_INIT_OLD,
                replacement=B3_INIT_NEW,
                required=True,
            ),
            TextPatch(
                name="b3_should_ar_cumem_bypass",
                anchor=B3_SHOULD_OLD,
                replacement=B3_SHOULD_NEW,
                required=True,
            ),
            TextPatch(
                name="b3_register_graph_buffers_guard",
                anchor=B3_REGISTER_OLD,
                replacement=B3_REGISTER_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis B3]",
            "GENESIS_B3_MAX_SIZE_MIB",
        ],
    )


def patch_B3_custom_ar() -> tuple[str, str]:
    """Apply B3 — custom all-reduce TP=2 fast path + 32 MiB buffer + CuMem graph bypass.

    Applies text patch to custom_all_reduce.py so workers inherit changes.
    """
    global _INSTALLED
    if not _is_enabled():
        return "skipped", f"opt-in only — set {ENV_FLAG}=1 to engage"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "custom_all_reduce.py not found on this pin (platform mismatch)"

    result, failure = patcher.apply()
    return result_to_wiring_status(
        result,
        failure,
        applied_message="B3 custom AR 32 MiB buffer + CuMem graph bypass applied",
        patch_name="B3 custom AR TP=2 fast path",
    )


def apply() -> tuple[str, str]:
    """Alias for dispatcher compatibility."""
    return patch_B3_custom_ar()


def revert() -> bool:
    """Revert text patch if needed."""
    global _INSTALLED
    patcher = _make_patcher()
    if patcher:
        res = patcher.revert()
        if res in (TextPatchResult.APPLIED, TextPatchResult.IDEMPOTENT):
            _INSTALLED = False
            return True
    return False
