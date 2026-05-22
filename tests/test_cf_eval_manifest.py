"""Tests for ``scripts/training/build_grpo_cf_eval_manifest.py``.

The manifest builder must produce an episode-stratified held-out CF eval slice
that is **disjoint** from the train pool, using the same split logic as
``run_grpo.py``. These tests pin that behavior so leakage cannot regress.

We synthesize a tiny activations shard with a handful of episodes and a paired
pairs JSONL, then invoke the builder as a subprocess (matches operator
workflow). Both the eval-pairs JSONL and the train/eval manifest pairs are
checked.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from nla.extraction import ActivationShardWriter, RunManifest


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER = REPO_ROOT / "scripts" / "training" / "build_grpo_cf_eval_manifest.py"

# Use the activation dim of the production extraction (the writer checks it
# against the manifest). Tests only need a couple of rows; tensor content is
# irrelevant.
_HIDDEN = 2048


def _write_tiny_shard(root: Path, n_episodes: int, steps_per_ep: int) -> list[str]:
    manifest = RunManifest(
        schema_version=1, model_repo="fake", layer_module_path="fake",
        hidden_size=_HIDDEN, activation_dtype="float32",
        embodiment_tag="FAKE",
    )
    writer = ActivationShardWriter(root, manifest, max_examples_per_shard=64)
    example_ids: list[str] = []
    T = 8
    for ep in range(n_episodes):
        for step in range(steps_per_ep):
            eid = f"goal__traj{ep:06d}_step{step:06d}"
            example_ids.append(eid)
            writer.write(
                example_id=eid,
                features=torch.zeros(T, _HIDDEN),
                attention_mask=torch.ones(T, dtype=torch.bool),
                image_mask=torch.zeros(T, dtype=torch.bool),
                input_ids=torch.zeros(T, dtype=torch.long),
                episode_index=ep, step_index=step,
                task_text="put the bowl on the plate", embodiment_tag="FAKE",
            )
    writer.close()
    return example_ids


def _write_pairs(path: Path, example_ids: list[str], *, suite: str = "goal") -> None:
    rows = []
    for eid in example_ids:
        rows.append({
            "source_example_id": eid,
            "example_id": f"{eid}@p001_anchor",
            "target_intent": "put the bowl on top of the cabinet",
            "target_task": "put_the_bowl_on_top_of_the_cabinet",
            "target_env_name": "libero_sim/put_the_bowl_on_top_of_the_cabinet",
            "source_intent": "put the bowl on the plate",
            "source_task": "put_the_bowl_on_the_plate",
            "is_counterfactual": True,
            "suite": suite,
        })
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _run_builder(*args: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(BUILDER), *args]
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )


def test_eval_manifest_is_disjoint_from_train(tmp_path: Path) -> None:
    act_root = tmp_path / "acts"
    pairs = tmp_path / "pairs.jsonl"
    out_prefix = tmp_path / "cf_eval"

    # 10 episodes × 4 steps = 40 activations; 5% held-out → 1 episode held out.
    ids = _write_tiny_shard(act_root, n_episodes=10, steps_per_ep=4)
    _write_pairs(pairs, ids)

    res = _run_builder(
        "--pairs", str(pairs),
        "--activations-root", str(act_root),
        "--out", str(out_prefix),
        "--seed", "0",
        "--held-out-fraction", "0.2",  # 2 of 10 episodes held out
        "--split-by", "episode",
    )
    assert res.returncode == 0, res.stderr

    eval_manifest = json.loads(
        (out_prefix.parent / (out_prefix.name + "_eval_manifest.json")).read_text()
    )
    train_manifest = json.loads(
        (out_prefix.parent / (out_prefix.name + "_train_manifest.json")).read_text()
    )

    eval_ids = set(eval_manifest["example_ids"])
    train_ids = set(train_manifest["example_ids"])
    assert eval_ids
    assert train_ids
    assert eval_ids.isdisjoint(train_ids), (
        f"eval and train manifests must be disjoint; overlap="
        f"{eval_ids & train_ids}"
    )
    assert eval_ids | train_ids == set(ids), (
        "manifests must cover every example_id"
    )

    eval_pairs_path = (
        out_prefix.parent / (out_prefix.name + "_pairs.jsonl")
    )
    eval_pair_ids: set[str] = set()
    with eval_pairs_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            eval_pair_ids.add(row["source_example_id"])
    assert eval_pair_ids, "eval pairs jsonl should not be empty"
    assert eval_pair_ids <= eval_ids, (
        "every eval pair row's source_example_id must be in the held-out "
        "manifest"
    )


def test_slice_filter_goal_only(tmp_path: Path) -> None:
    act_root = tmp_path / "acts"
    pairs = tmp_path / "pairs.jsonl"
    out_prefix = tmp_path / "cf_eval_goal"

    # Tag half of episodes with non-goal prefixes by re-writing pair rows.
    ids = _write_tiny_shard(act_root, n_episodes=8, steps_per_ep=2)
    rows = []
    for i, eid in enumerate(ids):
        # Force half the pair rows to claim a non-goal source (different
        # prefix) so the goal_only filter has something to drop.
        prefix = "goal__" if i % 2 == 0 else "spatial__"
        rows.append({
            "source_example_id": prefix + eid.split("__", 1)[1],
            "target_intent": "x",
            "target_task": "put_the_bowl_on_the_plate",
            "target_env_name": "libero_sim/put_the_bowl_on_the_plate",
            "is_counterfactual": True,
        })
    pairs.write_text("".join(json.dumps(r) + "\n" for r in rows))

    res = _run_builder(
        "--pairs", str(pairs),
        "--activations-root", str(act_root),
        "--out", str(out_prefix),
        "--seed", "0",
        "--held-out-fraction", "0.5",
        "--split-by", "episode",
        "--slice", "goal_only",
    )
    # Pairs reference IDs not in the activations root (the spatial__ ones), so
    # the builder filters those out via the "dropped_unknown_id" path. The
    # goal_only filter still requires goal__ prefix, so even matched IDs are
    # subject to it. Builder should still succeed.
    assert res.returncode == 0, res.stderr

    pairs_out = (out_prefix.parent / (out_prefix.name + "_pairs.jsonl"))
    n_rows = 0
    n_goal_only = 0
    if pairs_out.exists():
        with pairs_out.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_rows += 1
                row = json.loads(line)
                if str(row["source_example_id"]).startswith("goal__"):
                    n_goal_only += 1
    # Every kept row must be goal__; spatial__ rows are filtered.
    assert n_goal_only == n_rows


def test_no_pair_for_held_out_returns_empty(tmp_path: Path) -> None:
    """Empty pairs file is allowed; builder writes empty eval pairs jsonl."""
    act_root = tmp_path / "acts"
    pairs = tmp_path / "pairs.jsonl"
    out_prefix = tmp_path / "cf_eval_empty"

    _write_tiny_shard(act_root, n_episodes=4, steps_per_ep=2)
    pairs.write_text("")  # no rows

    res = _run_builder(
        "--pairs", str(pairs),
        "--activations-root", str(act_root),
        "--out", str(out_prefix),
        "--seed", "0",
        "--held-out-fraction", "0.5",
        "--split-by", "episode",
    )
    assert res.returncode == 0, res.stderr
    pairs_out = (out_prefix.parent / (out_prefix.name + "_pairs.jsonl"))
    assert pairs_out.exists()
    assert pairs_out.read_text() == ""
