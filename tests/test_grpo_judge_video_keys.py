"""Tests for ``--judge-video-keys`` plumbing in ``src/nla/training/grpo.py``.

GRPO's ``_image_paths_for_source`` is the load-bearing helper for the
multimodal-judge reward path; if it silently returns ``[]`` then the judge
short-circuits and every rollout receives ``r_judge = 0``, gutting the
reward signal. These tests assert:

1. The helper honors the new ``video_keys`` list (no hardcoded DROID
   suffixes any more).
2. ``GRPOConfig.judge_video_keys`` is a non-default field that travels
   through ``_validate_judge_config``.
3. ``_validate_judge_config`` fails loudly when the judge term is on but
   ``judge_video_keys`` is empty, even if ``frames_cache`` is set.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nla.training.grpo import (
    GRPOConfig,
    _image_paths_for_source,
    _validate_judge_config,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


def test_libero_keys_only(tmp_path: Path):
    """LIBERO is the canonical default: ``image`` + ``wrist_image`` resolve cleanly."""
    for name in (
        "src1__image.jpg",
        "src1__wrist_image.jpg",
        "src1__front_cam.jpg",  # unrelated key sitting in the cache; must be ignored.
    ):
        _touch(tmp_path / name)
    paths = _image_paths_for_source("src1", tmp_path, ["image", "wrist_image"])
    got = {Path(p).name for p in paths}
    assert got == {"src1__image.jpg", "src1__wrist_image.jpg"}, got


def test_alternate_keys_invariant(tmp_path: Path):
    """No hardcoded key list: arbitrary tokens supplied by the caller work too.

    Regression guard: if someone reintroduces a hardcoded suffix tuple, the
    LIBERO test above would still pass but this one would fail because the
    helper would no longer honor caller-supplied non-LIBERO tokens.
    """
    _touch(tmp_path / "src1__front_cam.jpg")
    _touch(tmp_path / "src1__side_cam.jpg")
    paths = _image_paths_for_source("src1", tmp_path, ["front_cam", "side_cam"])
    got = {Path(p).name for p in paths}
    assert got == {"src1__front_cam.jpg", "src1__side_cam.jpg"}, got


def test_missing_files_silent(tmp_path: Path):
    """Caller treats an empty list as ``r_judge = 0`` — never a hard raise."""
    _touch(tmp_path / "src1__image.jpg")  # only one of two requested keys
    paths = _image_paths_for_source("src1", tmp_path, ["image", "wrist_image"])
    assert paths == [str(tmp_path / "src1__image.jpg")], paths


def test_empty_keys_returns_empty(tmp_path: Path):
    _touch(tmp_path / "src1__image.jpg")
    assert _image_paths_for_source("src1", tmp_path, []) == []


def test_config_default_is_empty_list():
    cfg = GRPOConfig(sft_dir="x", activations_root="y", output_dir="z")
    assert cfg.judge_video_keys == []
    # The judge being off must not trip the new requirement either.
    _validate_judge_config(cfg)


def test_validate_requires_video_keys_when_judge_on(monkeypatch):
    """Judge enabled + frames_cache set but no video_keys -> hard ValueError."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")  # isolate from API-key check
    cfg = GRPOConfig(
        sft_dir="x", activations_root="y", output_dir="z",
        judge_reward_weight=0.5,
        frames_cache="/tmp/whatever",
    )
    with pytest.raises(ValueError, match="judge.video.keys|judge_video_keys"):
        _validate_judge_config(cfg)


def test_validate_passes_with_video_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = GRPOConfig(
        sft_dir="x", activations_root="y", output_dir="z",
        judge_reward_weight=0.5,
        frames_cache="/tmp/whatever",
        judge_video_keys=["image", "wrist_image"],
    )
    _validate_judge_config(cfg)  # must not raise
