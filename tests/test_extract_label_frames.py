"""Smoke + monkeypatch tests for ``scripts/eval/extract_label_frames.py``.

The script is glue: it iterates label rows, opens an ``EpisodeFrameLoader``
per episode, and writes ``{frames_cache}/{source_id}__{video_key}.jpg`` for
every key. We stub the loader so the test doesn't need a real LeRobot
dataset on disk and assert:

1. One JPEG is written per (source_id, video_key) pair.
2. Pre-existing files are reused (idempotent re-runs cost zero decode calls).
3. The script never tries to decode a frame for a row whose meta is missing
   the (episode_index, step_index) keys (the original libero pilot loader
   tolerated this; the extractor must keep tolerating it).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACT_PATH = REPO_ROOT / "scripts" / "eval" / "extract_label_frames.py"


def _load_extractor_module():
    name = "nla_extract_label_frames_test_alias"
    spec = importlib.util.spec_from_file_location(name, EXTRACT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _StubLoader:
    """Stands in for nla.labeling.frames.EpisodeFrameLoader.

    Records every (video_key, step_index) it's asked to decode so tests can
    assert the script does NOT re-decode existing files.
    """

    instances: list["_StubLoader"] = []

    def __init__(self, dataset_root: Path, episode_index: int):
        self.dataset_root = Path(dataset_root)
        self.episode_index = int(episode_index)
        self.decode_calls: list[tuple[str, int]] = []
        self.closed = False
        _StubLoader.instances.append(self)

    def frame(self, video_key: str, step_index: int) -> np.ndarray:
        self.decode_calls.append((video_key, int(step_index)))
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


def _stub_save_jpeg(frame: np.ndarray, path) -> Path:
    """Write a tiny placeholder so subsequent runs see the file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"jpg")
    return p


def _write_labels(labels_path: Path, rows: list[dict]) -> None:
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture(autouse=True)
def _reset_loader_instances():
    _StubLoader.instances.clear()
    yield
    _StubLoader.instances.clear()


def test_extractor_writes_one_jpeg_per_key(tmp_path: Path, monkeypatch, capsys):
    mod = _load_extractor_module()
    monkeypatch.setattr(mod, "EpisodeFrameLoader", _StubLoader)
    monkeypatch.setattr(mod, "save_jpeg", _stub_save_jpeg)

    dataset_root = tmp_path / "ds"
    dataset_root.mkdir()
    labels_path = tmp_path / "labels.jsonl"
    frames_cache = tmp_path / "frames_cache"

    _write_labels(labels_path, [
        {
            "example_id": "row_0",
            "description": "Robot reaches for the red cube.",
            "meta": {
                "episode_index": 0,
                "step_index": 5,
                "source_example_id": "traj000000_step000005",
                "position_type": "anchor",
            },
        },
        {
            "example_id": "row_1",
            "description": "Robot closes the gripper.",
            "meta": {
                "episode_index": 0,
                "step_index": 12,
                "source_example_id": "traj000000_step000012",
                "position_type": "image_patch",
            },
        },
    ])

    rc = mod.main([
        "--dataset-root", str(dataset_root),
        "--labels-jsonl", str(labels_path),
        "--frames-cache", str(frames_cache),
        "--video-keys", "image", "wrist_image",
    ])
    assert rc == 0

    expected = {
        "traj000000_step000005__image.jpg",
        "traj000000_step000005__wrist_image.jpg",
        "traj000000_step000012__image.jpg",
        "traj000000_step000012__wrist_image.jpg",
    }
    got = {p.name for p in frames_cache.iterdir()}
    assert got == expected, got

    # One loader per episode, reused across both rows.
    assert len(_StubLoader.instances) == 1
    loader = _StubLoader.instances[0]
    assert loader.episode_index == 0
    # 2 keys * 2 source_ids = 4 decode calls on a fresh cache.
    assert len(loader.decode_calls) == 4, loader.decode_calls
    # Loader was closed.
    assert loader.closed

    summary = capsys.readouterr().out
    assert "sources=2" in summary, summary
    assert "written=4" in summary, summary


def test_extractor_reuses_existing_files(tmp_path: Path, monkeypatch, capsys):
    mod = _load_extractor_module()
    monkeypatch.setattr(mod, "EpisodeFrameLoader", _StubLoader)
    monkeypatch.setattr(mod, "save_jpeg", _stub_save_jpeg)

    dataset_root = tmp_path / "ds"
    dataset_root.mkdir()
    labels_path = tmp_path / "labels.jsonl"
    frames_cache = tmp_path / "frames_cache"
    frames_cache.mkdir()
    # Pre-seed one of the two keys for one source.
    (frames_cache / "traj000000_step000005__image.jpg").write_bytes(b"existing")

    _write_labels(labels_path, [
        {
            "example_id": "row_0",
            "description": "ok",
            "meta": {
                "episode_index": 0,
                "step_index": 5,
                "source_example_id": "traj000000_step000005",
                "position_type": "anchor",
            },
        },
    ])

    rc = mod.main([
        "--dataset-root", str(dataset_root),
        "--labels-jsonl", str(labels_path),
        "--frames-cache", str(frames_cache),
        "--video-keys", "image", "wrist_image",
    ])
    assert rc == 0

    assert (frames_cache / "traj000000_step000005__image.jpg").read_bytes() == b"existing"
    assert (frames_cache / "traj000000_step000005__wrist_image.jpg").exists()

    # We only decoded the missing key.
    assert len(_StubLoader.instances) == 1
    assert _StubLoader.instances[0].decode_calls == [("wrist_image", 5)]

    summary = capsys.readouterr().out
    assert "written=1" in summary, summary
    assert "reused=1" in summary, summary


def test_extractor_skips_rows_without_episode_step(tmp_path: Path, monkeypatch, capsys):
    mod = _load_extractor_module()
    monkeypatch.setattr(mod, "EpisodeFrameLoader", _StubLoader)
    monkeypatch.setattr(mod, "save_jpeg", _stub_save_jpeg)

    dataset_root = tmp_path / "ds"
    dataset_root.mkdir()
    labels_path = tmp_path / "labels.jsonl"
    frames_cache = tmp_path / "frames_cache"

    _write_labels(labels_path, [
        # Bad row: missing episode_index.
        {
            "example_id": "row_0",
            "description": "ok",
            "meta": {"step_index": 1, "position_type": "anchor"},
        },
        # Good row.
        {
            "example_id": "row_1",
            "description": "ok",
            "meta": {
                "episode_index": 1,
                "step_index": 3,
                "source_example_id": "traj000001_step000003",
                "position_type": "anchor",
            },
        },
        # Error row: skip.
        {
            "example_id": "row_2",
            "error": "labeler raised",
            "meta": {"episode_index": 1, "step_index": 4},
        },
    ])

    rc = mod.main([
        "--dataset-root", str(dataset_root),
        "--labels-jsonl", str(labels_path),
        "--frames-cache", str(frames_cache),
        "--video-keys", "image",
    ])
    assert rc == 0
    got = {p.name for p in frames_cache.iterdir()}
    assert got == {"traj000001_step000003__image.jpg"}, got

    summary = capsys.readouterr().out
    assert "sources=1" in summary
