"""Unit tests for ``scripts/labeling/audit_frame_sync.py``.

These exercise the alignment-classification logic end-to-end against
synthetic ``index.jsonl`` / ``labels.jsonl`` / ``frames_cache`` directories
inside ``tmp_path``. No real JPEG decoding is required: cached "frames"
are zero-byte placeholder files and we drive the sidecar-based path.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "labeling" / "audit_frame_sync.py"


def _load_script_module():
    """Import the script-under-test as a module without polluting other tests."""
    spec = importlib.util.spec_from_file_location("audit_frame_sync_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def afs():
    return _load_script_module()


VIEWS = ("image", "wrist_image")


def _write_index(activations_root: Path, records: list[dict]) -> None:
    activations_root.mkdir(parents=True, exist_ok=True)
    with (activations_root / "index.jsonl").open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _write_labels(labels_jsonl: Path, rows: list[dict]) -> None:
    labels_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with labels_jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _make_frames(frames_cache: Path, source_id: str, *, sidecar: dict | None = None) -> None:
    frames_cache.mkdir(parents=True, exist_ok=True)
    for vk in VIEWS:
        (frames_cache / f"{source_id}__{vk}.jpg").write_bytes(b"")
        if sidecar is not None:
            (frames_cache / f"{source_id}__{vk}.json").write_text(json.dumps(sidecar))


def _record(example_id: str, ep: int, st: int) -> dict:
    return {
        "example_id": example_id,
        "shard_id": 0,
        "local_index": 0,
        "seq_len": 277,
        "task_index": None,
        "task_text": None,
        "episode_index": ep,
        "step_index": st,
        "embodiment_tag": "test",
        "extra": {},
    }


def _label_row(source_id: str, ep: int, st: int, position_index: int = 0) -> dict:
    return {
        "example_id": f"{source_id}@p{position_index:03d}_image_patch",
        "description": "stub",
        "model": "test",
        "kind": "position",
        "meta": {
            "position_index": position_index,
            "position_type": "image_patch",
            "seq_len": 277,
            "episode_index": ep,
            "step_index": st,
            "instruction": "do the thing",
            "source_example_id": source_id,
        },
    }


def _setup_basic(tmp_path: Path, *, n: int = 4, sidecar_step_offset: int | None = 0,
                 with_sidecar: bool = True, missing_frame_for: list[int] | None = None,
                 off_by_one_indices: set[int] | None = None) -> dict:
    """Return paths for a synthetic dataset with N labeled examples."""
    activations = tmp_path / "activations"
    labels_jsonl = tmp_path / "labels" / "labels.jsonl"
    frames_cache = tmp_path / "labels" / "frames_cache"
    out_json = tmp_path / "audit.json"

    records, rows = [], []
    for i in range(n):
        source_id = f"traj{i+1:06d}_step{i+10:06d}"
        ep, st = i + 1, i + 10
        records.append(_record(source_id, ep, st))
        rows.append(_label_row(source_id, ep, st, position_index=i))

        if missing_frame_for and i in missing_frame_for:
            continue
        sidecar = None
        if with_sidecar:
            sidecar_st = st + (1 if (off_by_one_indices and i in off_by_one_indices) else 0)
            sidecar = {"episode_index": ep, "step_index": sidecar_st}
        _make_frames(frames_cache, source_id, sidecar=sidecar)

    _write_index(activations, records)
    _write_labels(labels_jsonl, rows)
    return {
        "activations": activations,
        "labels_jsonl": labels_jsonl,
        "frames_cache": frames_cache,
        "out_json": out_json,
        "n": n,
    }


def _run_main(afs, paths: dict, *, strict: bool = False, extra: list[str] | None = None) -> int:
    argv = [
        "--labels-jsonl", str(paths["labels_jsonl"]),
        "--activations-root", str(paths["activations"]),
        "--frames-cache", str(paths["frames_cache"]),
        "--n-sample", str(paths["n"]),
        "--seed", "0",
        "--out-json", str(paths["out_json"]),
        "--no-pixel-diff",
    ]
    if strict:
        argv.append("--strict")
    if extra:
        argv += extra
    return afs.main(argv)


def test_aligned_count(tmp_path, afs):
    paths = _setup_basic(tmp_path, n=5, with_sidecar=True)
    code = _run_main(afs, paths)
    assert code == 0
    summary = json.loads(paths["out_json"].read_text())["summary"]
    assert summary["n_total"] == 5
    assert summary["n_aligned"] == 5
    assert summary["n_off_by_one"] == 0
    assert summary["n_missing_frame"] == 0
    assert summary["n_unverifiable"] == 0


def test_off_by_one_detected(tmp_path, afs):
    # 3 aligned, 1 off-by-one (index 2).
    paths = _setup_basic(tmp_path, n=4, with_sidecar=True, off_by_one_indices={2})
    code = _run_main(afs, paths)
    summary = json.loads(paths["out_json"].read_text())["summary"]
    assert summary["n_off_by_one"] == 1
    assert summary["n_aligned"] == 3
    # 3/4 = 0.75 < 0.99 → exit 1 in non-strict mode.
    assert code == 1


def test_missing_frame_counted(tmp_path, afs):
    # Index 1 has no JPEGs; the rest are aligned via sidecar.
    paths = _setup_basic(tmp_path, n=3, with_sidecar=True, missing_frame_for=[1])
    code = _run_main(afs, paths)
    summary = json.loads(paths["out_json"].read_text())["summary"]
    assert summary["n_missing_frame"] == 1
    assert summary["n_aligned"] == 2
    assert code == 1


def test_no_sidecar_unverifiable(tmp_path, afs, capsys):
    paths = _setup_basic(tmp_path, n=4, with_sidecar=False)
    code = _run_main(afs, paths)
    summary = json.loads(paths["out_json"].read_text())["summary"]
    assert summary["n_unverifiable"] == summary["n_total"] == 4
    assert summary["n_aligned"] == 0
    assert code == 0
    out = capsys.readouterr().out
    assert "warning" in out.lower()


def test_strict_mode_off_by_one_exits_nonzero(tmp_path, afs):
    paths = _setup_basic(tmp_path, n=4, with_sidecar=True, off_by_one_indices={1})
    code = _run_main(afs, paths, strict=True)
    summary = json.loads(paths["out_json"].read_text())["summary"]
    assert summary["n_off_by_one"] == 1
    assert code == 1


def test_strict_mode_aligned_exits_zero(tmp_path, afs):
    paths = _setup_basic(tmp_path, n=3, with_sidecar=True)
    code = _run_main(afs, paths, strict=True)
    assert code == 0
