"""Unit tests for ``CounterfactualPairSampler`` dual-key indexing + multi-path merge.

The mining pipeline writes both ``source_example_id`` (the activation-shard id
the trainer's :class:`SampledPositionDataset` carries) and ``example_id`` (the
label-row id). Datasets that emit the latter used to silently miss every CF
pair because the old sampler indexed only by ``source_example_id``. These
tests pin the new behavior: either key resolves to the same candidate list,
and rows are deduped per id-bucket so a pair appearing under both keys (or in
multiple files) isn't double-weighted by ``random.choice``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nla.training.counterfactual_data import CounterfactualPairSampler


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_sampler_indexes_by_source_id_and_example_id(tmp_path: Path):
    p = tmp_path / "pairs.jsonl"
    _write_jsonl(p, [
        {
            "source_example_id": "src_alpha",
            "example_id": "lbl_alpha",
            "target_intent": "put the bowl on the plate",
            "target_task": "put_the_bowl_on_the_plate",
            "target_env_name": "LIBERO_GOAL_put_the_bowl_on_the_plate",
            "is_counterfactual": False,
        },
    ])
    s = CounterfactualPairSampler(p, seed=0, validate_bodies_in_bddl=False)
    # Both keys resolve to the same single candidate pair.
    assert s.has("src_alpha")
    assert s.has("lbl_alpha")
    via_src = s.sample_for(["src_alpha"])[0]
    via_lbl = s.sample_for(["lbl_alpha"])[0]
    assert via_src.target_intent == via_lbl.target_intent == "put the bowl on the plate"
    assert via_src.target_task == via_lbl.target_task


def test_sampler_dedups_when_source_id_equals_example_id(tmp_path: Path):
    # A row where ``example_id == source_example_id`` should be stored
    # exactly once under that key (no double-weighting of the same pair).
    p = tmp_path / "pairs.jsonl"
    _write_jsonl(p, [
        {
            "source_example_id": "same_id",
            "example_id": "same_id",
            "target_intent": "do A",
            "target_task": "A",
            "target_env_name": "env_A",
        },
        {
            "source_example_id": "same_id",
            "example_id": "same_id",
            "target_intent": "do B",
            "target_task": "B",
            "target_env_name": "env_B",
        },
    ])
    s = CounterfactualPairSampler(p, seed=0, validate_bodies_in_bddl=False)
    # Internal bucket has exactly the two distinct pairs (not 4).
    assert len(s._by_id["same_id"]) == 2


def test_sampler_dedups_same_pair_across_multiple_files(tmp_path: Path):
    # A pair appearing in two files (e.g. an overlap slice) should not
    # be counted twice under the same id.
    p1 = tmp_path / "primary.jsonl"
    p2 = tmp_path / "extra.jsonl"
    common = {
        "source_example_id": "src_dup",
        "example_id": "src_dup",
        "target_intent": "do the dup task",
        "target_task": "dup_task",
        "target_env_name": "env_dup",
    }
    _write_jsonl(p1, [common])
    _write_jsonl(p2, [common])
    s = CounterfactualPairSampler(p1, seed=0, additional_paths=[p2], validate_bodies_in_bddl=False)
    assert len(s._by_id["src_dup"]) == 1


def test_sampler_merges_distinct_pairs_across_files(tmp_path: Path):
    p1 = tmp_path / "primary.jsonl"
    p2 = tmp_path / "extra.jsonl"
    _write_jsonl(p1, [
        {
            "source_example_id": "src_x",
            "example_id": "lbl_x",
            "target_intent": "A",
            "target_task": "tA",
            "target_env_name": "envA",
        },
    ])
    _write_jsonl(p2, [
        {
            "source_example_id": "src_x",
            "example_id": "lbl_x",
            "target_intent": "B",
            "target_task": "tB",
            "target_env_name": "envB",
        },
    ])
    s = CounterfactualPairSampler(p1, seed=0, additional_paths=[p2], validate_bodies_in_bddl=False)
    assert len(s._by_id["src_x"]) == 2
    assert len(s._by_id["lbl_x"]) == 2


def test_sampler_falls_back_for_unknown_id(tmp_path: Path):
    p = tmp_path / "pairs.jsonl"
    _write_jsonl(p, [
        {
            "source_example_id": "known",
            "target_intent": "x",
            "target_task": "tx",
            "target_env_name": "envx",
        },
    ])
    s = CounterfactualPairSampler(
        p, seed=0,
        fallback_intent="(none)",
        fallback_env_name="(no_env)",
    )
    out = s.sample_for(["unknown_id"])
    assert out[0].target_intent == "(none)"
    assert out[0].target_task == ""
    assert out[0].target_env_name == "(no_env)"


def test_sampler_raises_on_missing_extra_path(tmp_path: Path):
    p = tmp_path / "pairs.jsonl"
    _write_jsonl(p, [
        {
            "source_example_id": "s",
            "target_intent": "x",
            "target_task": "t",
            "target_env_name": "e",
        },
    ])
    with pytest.raises(FileNotFoundError):
        CounterfactualPairSampler(
            p, seed=0, additional_paths=[tmp_path / "nope.jsonl"],
            validate_bodies_in_bddl=False,
        )


def test_sampler_skips_example_id_when_only_source_present(tmp_path: Path):
    # Older miner runs only wrote source_example_id. Confirm the sampler
    # still loads them and indexes under that single key.
    p = tmp_path / "pairs.jsonl"
    _write_jsonl(p, [
        {
            "source_example_id": "old_style",
            "target_intent": "y",
            "target_task": "ty",
            "target_env_name": "envy",
        },
    ])
    s = CounterfactualPairSampler(p, seed=0, validate_bodies_in_bddl=False)
    assert s.has("old_style")
    assert not s.has("anything_else")


def test_sampler_skips_rows_with_missing_bodies_when_validating(tmp_path: Path):
    p = tmp_path / "pairs.jsonl"
    _write_jsonl(p, [
        {
            "source_example_id": "src_bad",
            "target_intent": "put the wine bottle on the rack",
            "target_task": "put_the_wine_bottle_on_the_rack",
            "target_env_name": "libero_sim/put_the_wine_bottle_on_the_rack",
        },
    ])
    bddl_dir = tmp_path / "bddl"
    bddl_dir.mkdir()
    (bddl_dir / "put_the_wine_bottle_on_the_rack.bddl").write_text(
        "(:objects\n  plate_1 - plate\n)\n"
    )
    s = CounterfactualPairSampler(
        p, seed=0, validate_bodies_in_bddl=True, bddl_dir=bddl_dir,
    )
    assert not s.has("src_bad")


def test_collect_cf_eligible_example_ids(tmp_path: Path):
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    _write_jsonl(p1, [
        {"source_example_id": "goal__traj000001_step000002"},
        {"source_example_id": "spatial__traj000010_step000004"},
    ])
    _write_jsonl(p2, [
        {"source_example_id": "spatial__traj000010_step000004"},
        {"source_example_id": "10__traj000099_step000000"},
    ])
    from nla.training.counterfactual_data import (
        collect_cf_eligible_example_ids,
        load_grpo_cf_manifest,
        MANIFEST_VERSION,
    )

    ids = collect_cf_eligible_example_ids([p1, p2])
    assert ids == {
        "goal__traj000001_step000002",
        "spatial__traj000010_step000004",
        "10__traj000099_step000000",
    }

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "version": MANIFEST_VERSION,
        "example_ids": sorted(ids),
    }))
    assert load_grpo_cf_manifest(manifest_path) == ids
