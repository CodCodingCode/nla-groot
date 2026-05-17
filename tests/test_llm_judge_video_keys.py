"""Tests for ``--video-keys`` plumbing in ``scripts/eval/llm_judge_av_captions.py``.

The script's ``_image_paths_for`` is the single helper that maps a label's
``source_example_id`` plus a list of camera-key tokens onto JPEG paths in a
flat frames cache. These tests make sure:

1. Only files matching the requested keys are returned (so a stale DROID-style
   cache cannot leak into a LIBERO eval and vice-versa).
2. The exact filename pattern is ``{source_id}__{video_key}.jpg`` (so we
   don't regress to the old hardcoded suffix list).
3. Missing files are skipped silently, not raised; the caller is responsible
   for dropping rows that resolve to an empty list.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
JUDGE_PATH = REPO_ROOT / "scripts" / "eval" / "llm_judge_av_captions.py"


def _load_judge_module():
    """Importlib-load the script so we can call ``_image_paths_for`` directly.

    The script has heavy optional imports (transformers, torch) at module
    scope, but ``_image_paths_for`` itself is pure-filesystem. We must
    register the module in ``sys.modules`` before exec'ing so any internal
    relative references resolve cleanly.
    """
    name = "nla_llm_judge_av_captions_test_alias"
    spec = importlib.util.spec_from_file_location(name, JUDGE_PATH)
    assert spec and spec.loader, f"failed to load {JUDGE_PATH}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


def test_libero_keys_only(tmp_path: Path):
    """LIBERO is the canonical default: ``image`` + ``wrist_image`` resolve cleanly."""
    mod = _load_judge_module()
    _touch(tmp_path / "src1__image.jpg")
    _touch(tmp_path / "src1__wrist_image.jpg")
    # Same row also has an unrelated key sitting in the cache; must be ignored.
    _touch(tmp_path / "src1__front_cam.jpg")

    paths = mod._image_paths_for("src1", tmp_path, ["image", "wrist_image"])
    got = {Path(p).name for p in paths}
    assert got == {"src1__image.jpg", "src1__wrist_image.jpg"}, got


def test_alternate_keys_invariant(tmp_path: Path):
    """The helper has no hardcoded key list: arbitrary tokens must resolve too.

    This guards the dataset-agnostic invariant — if someone reintroduces a
    DROID-style hardcoded suffix tuple, the LIBERO-style assertion above
    would still pass; this test would fail because the helper would no
    longer honor caller-supplied non-LIBERO tokens.
    """
    mod = _load_judge_module()
    _touch(tmp_path / "src1__front_cam.jpg")
    _touch(tmp_path / "src1__side_cam.jpg")
    _touch(tmp_path / "src1__image.jpg")  # LIBERO file present but not requested.

    paths = mod._image_paths_for("src1", tmp_path, ["front_cam", "side_cam"])
    got = {Path(p).name for p in paths}
    assert got == {"src1__front_cam.jpg", "src1__side_cam.jpg"}, got


def test_missing_key_silently_skipped(tmp_path: Path):
    """Only one of two requested keys exists -> the present one is returned."""
    mod = _load_judge_module()
    _touch(tmp_path / "src1__image.jpg")
    # wrist_image intentionally missing.

    paths = mod._image_paths_for("src1", tmp_path, ["image", "wrist_image"])
    assert paths == [str(tmp_path / "src1__image.jpg")], paths


def test_no_keys_returns_empty(tmp_path: Path):
    """No requested keys -> empty list, no raise."""
    mod = _load_judge_module()
    _touch(tmp_path / "src1__image.jpg")
    assert mod._image_paths_for("src1", tmp_path, []) == []


def test_unknown_source_returns_empty(tmp_path: Path):
    """source_id with no matching files -> empty list (caller drops the row)."""
    mod = _load_judge_module()
    _touch(tmp_path / "other__image.jpg")
    assert mod._image_paths_for("src1", tmp_path, ["image", "wrist_image"]) == []


def test_filename_pattern_is_double_underscore(tmp_path: Path):
    """Regression: the pattern must be ``__{key}.jpg``, not ``_{key}.jpg``
    nor ``{key}.jpg``. A single-underscore file is NOT a match."""
    mod = _load_judge_module()
    _touch(tmp_path / "src1__image.jpg")
    _touch(tmp_path / "src1_image.jpg")  # single underscore — must not match
    _touch(tmp_path / "image.jpg")        # no source prefix — must not match

    paths = mod._image_paths_for("src1", tmp_path, ["image"])
    assert paths == [str(tmp_path / "src1__image.jpg")], paths
