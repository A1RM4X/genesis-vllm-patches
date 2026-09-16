# SPDX-License-Identifier: Apache-2.0
"""TDD for PN114 — Mamba align bounds guard (vllm#35288 fix)."""
from __future__ import annotations

import pytest


def _wiring():
    from vllm._genesis.wiring.spec_decode import (
        patch_PN114_mamba_align_bounds_guard as M,
    )
    return M


def test_anchors_present():
    M = _wiring()
    assert "dest_block_id" in M.ANCHOR_OLD_1
    assert "CONV_STATE_DIM_FIRST" in M.ANCHOR_OLD_2
    assert "SD conv: copy" in M.ANCHOR_OLD_3
    assert "Temporal state: copy" in M.ANCHOR_OLD_4
    assert "Skip no-op self-copy." in M.ANCHOR_OLD_5


def test_replacements_contain_guards():
    M = _wiring()
    assert "src_col < 0 or dst_col < 0" in M.ANCHOR_NEW_1
    assert "dest_block_id < 0" in M.ANCHOR_NEW_1
    assert "src_block_id < 0" in M.ANCHOR_NEW_2
    assert "src_block_id < 0" in M.ANCHOR_NEW_3
    assert "actual_src_block_id < 0" in M.ANCHOR_NEW_4
    assert "src_block_idx < 0 or dest_block_idx < 0" in M.ANCHOR_NEW_5


def test_idempotent_on_synthetic(tmp_path):
    from vllm._genesis.wiring.text_patch import (
        TextPatch, TextPatcher, TextPatchResult,
    )
    M = _wiring()
    target = tmp_path / "mamba_utils.py"
    target.write_text(
        "# header\n"
        + M.ANCHOR_OLD_1 + "\n# mid 1\n"
        + M.ANCHOR_OLD_2 + "\n# mid 2\n"
        + M.ANCHOR_OLD_3 + "\n# mid 3\n"
        + M.ANCHOR_OLD_4 + "\n# mid 4\n"
        + M.ANCHOR_OLD_5 + "\n# tail\n"
    )
    patcher = TextPatcher(
        patch_name="PN114 test",
        target_file=str(target),
        marker=M.GENESIS_PN114_MARKER,
        sub_patches=[
            TextPatch(name="sub1", anchor=M.ANCHOR_OLD_1, replacement=M.ANCHOR_NEW_1, required=True),
            TextPatch(name="sub2", anchor=M.ANCHOR_OLD_2, replacement=M.ANCHOR_NEW_2, required=True),
            TextPatch(name="sub3", anchor=M.ANCHOR_OLD_3, replacement=M.ANCHOR_NEW_3, required=True),
            TextPatch(name="sub4", anchor=M.ANCHOR_OLD_4, replacement=M.ANCHOR_NEW_4, required=True),
            TextPatch(name="sub5", anchor=M.ANCHOR_OLD_5, replacement=M.ANCHOR_NEW_5, required=True),
        ],
    )
    r1, _ = patcher.apply()
    assert r1 == TextPatchResult.APPLIED
    body1 = target.read_text()
    assert "PN114" in body1
    assert "src_col < 0 or dst_col < 0" in body1
    r2, _ = patcher.apply()
    assert r2 == TextPatchResult.IDEMPOTENT


def test_env_flag_default_on(monkeypatch):
    from vllm._genesis.dispatcher import should_apply
    monkeypatch.delenv("GENESIS_ENABLE_PN114_MAMBA_ALIGN_BOUNDS_GUARD", raising=False)
    decision, _ = should_apply("PN114")
    assert decision is True


def test_env_flag_disables(monkeypatch):
    from vllm._genesis.dispatcher import should_apply
    monkeypatch.setenv("GENESIS_ENABLE_PN114_MAMBA_ALIGN_BOUNDS_GUARD", "0")
    decision, _ = should_apply("PN114")
    assert decision is False
