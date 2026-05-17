"""Unit tests for ``nla.training.replay_manifest``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from nla.extraction import ActivationShardWriter, RunManifest
from nla.training.replay_manifest import (
    ReplayEntry,
    ReplayManifest,
    build_replay_manifest,
    parse_example_id,
)


def test_parse_example_id_with_suite():
    suite, traj, step = parse_example_id("goal__traj000001_step000058")
    assert suite == "goal"
    assert (traj, step) == (1, 58)


def test_parse_example_id_without_suite():
    suite, traj, step = parse_example_id("traj000123_step000000")
    assert suite is None
    assert (traj, step) == (123, 0)


def test_parse_example_id_rejects_garbage():
    with pytest.raises(ValueError):
        parse_example_id("not_a_real_id")


def test_manifest_lookup_and_serialization(tmp_path: Path):
    entries = [
        ReplayEntry("goal__traj000000_step000000", "goal", 0, 0, "/tmp/goal"),
        ReplayEntry("goal__traj000000_step000002", "goal", 0, 2, "/tmp/goal"),
    ]
    m = ReplayManifest(entries)
    assert len(m) == 2
    assert "goal__traj000000_step000002" in m
    got = m.get("goal__traj000000_step000002")
    assert got is not None and got.traj_idx == 0 and got.step_idx == 2
    assert m.suites == ["goal"]

    out = tmp_path / "manifest.jsonl"
    m.save(out)
    m2 = ReplayManifest.load(out)
    assert [asdict_eq(a, b) for a, b in zip(m.entries, m2.entries)]


def asdict_eq(a: ReplayEntry, b: ReplayEntry) -> bool:
    return (a.example_id, a.suite, a.traj_idx, a.step_idx, a.dataset_root) == (
        b.example_id, b.suite, b.traj_idx, b.step_idx, b.dataset_root
    )


def test_manifest_duplicate_ids_raise():
    e = ReplayEntry("goal__traj000000_step000000", "goal", 0, 0, "/tmp/goal")
    with pytest.raises(ValueError, match="Duplicate"):
        ReplayManifest([e, e])


def _write_dump(out_root: Path, ids: list[str]) -> None:
    manifest = RunManifest(
        schema_version=1,
        model_repo="fake",
        layer_module_path="fake",
        hidden_size=16,
        activation_dtype="float32",
        embodiment_tag="FAKE",
    )
    writer = ActivationShardWriter(out_root, manifest, max_examples_per_shard=64)
    for eid in ids:
        writer.write(
            example_id=eid,
            features=torch.zeros(4, 16),
            attention_mask=torch.ones(4, dtype=torch.bool),
            image_mask=torch.zeros(4, dtype=torch.bool),
        )
    writer.close()


def test_build_replay_manifest_skips_unknown_suites_and_caches(tmp_path: Path):
    act_root = tmp_path / "acts"
    _write_dump(
        act_root,
        [
            "goal__traj000000_step000000",
            "goal__traj000000_step000002",
            "object__traj000001_step000010",
            "unparseable",
        ],
    )
    cache = tmp_path / "manifest.jsonl"
    m = build_replay_manifest(
        act_root,
        # No mapping for "object" -> rows for that suite are skipped.
        dataset_roots_by_suite={"goal": tmp_path / "datasets/goal"},
        cache_path=cache,
    )
    assert len(m) == 2
    ids = sorted(e.example_id for e in m)
    assert ids == ["goal__traj000000_step000000", "goal__traj000000_step000002"]
    assert cache.exists()
    # Each ReplayEntry resolves the dataset_root to an absolute path.
    for e in m:
        assert Path(e.dataset_root).is_absolute()

    # Reloading from cache should produce equivalent entries even when the
    # dataset map is now stale (cache wins to avoid rescanning).
    m2 = build_replay_manifest(
        act_root,
        dataset_roots_by_suite={"goal": tmp_path / "ignored"},
        cache_path=cache,
    )
    assert [e.example_id for e in m2] == [e.example_id for e in m]


def test_build_replay_manifest_raises_on_unparseable_when_strict(tmp_path: Path):
    act_root = tmp_path / "acts"
    _write_dump(act_root, ["definitely_not_traj_format"])
    with pytest.raises(ValueError):
        build_replay_manifest(
            act_root,
            dataset_roots_by_suite={None: tmp_path / "ds"},
            skip_unparseable=False,
        )
