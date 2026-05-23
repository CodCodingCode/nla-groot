"""Argparse / leakage-guard / dry-run tests for ``compare_cf_steer_checkpoints.py``.

These do **not** spin up GR00T, the steer server, or LIBERO. They exercise:

  - ``--dry-run`` short-circuits before any heavy import.
  - ``--exclude-ids-path`` correctly drops pair rows whose source_example_id
    appears in the train manifest.
  - ``--deterministic-order`` returns the file-order prefix (no shuffle).
  - ``--require-held-out`` causes a non-zero exit when leakage is forced via
    a degenerate exclude manifest.
  - Unknown intent/causal arms are rejected before any rollout.

The compare script's heavy ops (AV/AR load, sim worker) are gated behind the
dry-run check, so these tests run on CPU in CI without GPU or LIBERO.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARE = REPO_ROOT / "scripts" / "eval" / "compare_cf_steer_checkpoints.py"


def _fake_av(dir_: Path) -> Path:
    """Create the minimum file layout compare expects to consider an AV checkpoint."""
    av_dir = dir_ / "av"
    av_dir.mkdir(parents=True, exist_ok=True)
    (av_dir / "av_config.json").write_text("{}")
    return dir_


def _fake_ar(dir_: Path) -> None:
    ar_dir = dir_ / "ar"
    ar_dir.mkdir(parents=True, exist_ok=True)
    (ar_dir / "ar_config.json").write_text("{}")


def _make_sft(dir_: Path) -> Path:
    _fake_av(dir_)
    _fake_ar(dir_)
    return dir_


def _make_grpo_av(dir_: Path) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "av_config.json").write_text("{}")
    return dir_


def _write_pairs(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _write_manifest(path: Path, ids: list[str], kind: str = "train") -> None:
    path.write_text(json.dumps({
        "version": 1,
        "kind": kind,
        "example_ids": ids,
    }))


def _run(*args: str, expect_rc: int | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(COMPARE), *args]
    res = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": str(REPO_ROOT / "src"),
            "PATH": "/usr/bin:/bin",
        },
    )
    if expect_rc is not None:
        assert res.returncode == expect_rc, (
            f"expected rc={expect_rc} got {res.returncode}\nstdout={res.stdout}\nstderr={res.stderr}"
        )
    return res


def _row(sid: str, *, task: str = "put_the_bowl_on_the_plate") -> dict:
    return {
        "source_example_id": sid,
        "target_intent": "put the bowl on the plate",
        "target_task": task,
        "target_env_name": f"libero_sim/{task}",
        "source_intent": "put the wine bottle on the rack",
        "source_task": "put_the_wine_bottle_on_the_rack",
        "is_counterfactual": True,
    }


def test_dry_run_smoke(tmp_path: Path) -> None:
    sft = _make_sft(tmp_path / "sft")
    grpo_av = _make_grpo_av(tmp_path / "grpo_av")
    pairs = tmp_path / "pairs.jsonl"
    _write_pairs(pairs, [_row(f"goal__traj{i:06d}_step{i:06d}") for i in range(5)])
    out_json = tmp_path / "out.json"
    res = _run(
        "--sft-dir", str(sft),
        "--grpo-av-dir", str(grpo_av),
        "--pairs-path", str(pairs),
        "--activations-root", str(tmp_path / "acts"),
        "--out-json", str(out_json),
        "--n-samples", "3",
        "--dry-run",
        expect_rc=0,
    )
    assert "CF steer compare: 3 samples" in res.stdout


def test_exclude_ids_drops_train_ids(tmp_path: Path) -> None:
    sft = _make_sft(tmp_path / "sft")
    grpo_av = _make_grpo_av(tmp_path / "grpo_av")
    pairs = tmp_path / "pairs.jsonl"
    train_manifest = tmp_path / "train_manifest.json"
    train_ids = [f"goal__traj{i:06d}_step{i:06d}" for i in range(3)]
    eval_ids = [f"goal__traj{i:06d}_step{i:06d}" for i in range(3, 8)]
    _write_pairs(pairs, [_row(sid) for sid in train_ids + eval_ids])
    _write_manifest(train_manifest, train_ids, kind="train")
    out_json = tmp_path / "out.json"

    res = _run(
        "--sft-dir", str(sft),
        "--grpo-av-dir", str(grpo_av),
        "--pairs-path", str(pairs),
        "--activations-root", str(tmp_path / "acts"),
        "--out-json", str(out_json),
        "--exclude-ids-path", str(train_manifest),
        "--deterministic-order",
        "--require-held-out",
        "--n-samples", "10",
        "--dry-run",
        expect_rc=0,
    )
    # Only the eval_ids survive the exclude filter.
    selected = [
        line for line in res.stdout.splitlines()
        if line.lstrip().startswith("[")
        and "goal__traj" in line
        and "->" in line
    ]
    assert len(selected) == len(eval_ids), res.stdout
    for sid in eval_ids:
        assert sid in res.stdout
    for sid in train_ids:
        assert sid not in "\n".join(selected), (
            f"train id {sid} leaked into eval"
        )


def test_require_held_out_fails_when_all_ids_excluded(tmp_path: Path) -> None:
    sft = _make_sft(tmp_path / "sft")
    grpo_av = _make_grpo_av(tmp_path / "grpo_av")
    pairs = tmp_path / "pairs.jsonl"
    train_manifest = tmp_path / "train_manifest.json"
    ids = [f"goal__traj{i:06d}_step{i:06d}" for i in range(4)]
    _write_pairs(pairs, [_row(sid) for sid in ids])
    # Empty train manifest means no filtering. Use a path that allows the
    # exclude flag without dropping anything; then manually reuse-pairs-json
    # to force a leakage scenario.
    _write_manifest(train_manifest, ids, kind="train")
    reuse = tmp_path / "reuse.json"
    reuse.write_text(json.dumps({
        "samples": [
            {**_row(ids[0]), "position_index": 0, "position_type": "anchor"},
        ],
    }))
    out_json = tmp_path / "out.json"

    res = _run(
        "--sft-dir", str(sft),
        "--grpo-av-dir", str(grpo_av),
        "--pairs-path", str(pairs),
        "--activations-root", str(tmp_path / "acts"),
        "--out-json", str(out_json),
        "--exclude-ids-path", str(train_manifest),
        "--require-held-out",
        "--reuse-pairs-json", str(reuse),
        "--n-samples", "1",
    )
    # require_held_out must abort with rc=3 because the reused sample is in
    # the train manifest.
    assert res.returncode == 3, (res.stdout, res.stderr)


def test_unknown_intent_arm_rejected(tmp_path: Path) -> None:
    sft = _make_sft(tmp_path / "sft")
    grpo_av = _make_grpo_av(tmp_path / "grpo_av")
    pairs = tmp_path / "pairs.jsonl"
    _write_pairs(pairs, [_row("goal__traj000000_step000000")])
    res = _run(
        "--sft-dir", str(sft),
        "--grpo-av-dir", str(grpo_av),
        "--pairs-path", str(pairs),
        "--activations-root", str(tmp_path / "acts"),
        "--out-json", str(tmp_path / "out.json"),
        "--intent-arms", "bogus_arm",
        "--n-samples", "1",
    )
    assert res.returncode == 2, (res.stdout, res.stderr)
    assert "not in" in res.stderr


def test_unknown_causal_arm_rejected(tmp_path: Path) -> None:
    sft = _make_sft(tmp_path / "sft")
    grpo_av = _make_grpo_av(tmp_path / "grpo_av")
    pairs = tmp_path / "pairs.jsonl"
    _write_pairs(pairs, [_row("goal__traj000000_step000000")])
    res = _run(
        "--sft-dir", str(sft),
        "--grpo-av-dir", str(grpo_av),
        "--pairs-path", str(pairs),
        "--activations-root", str(tmp_path / "acts"),
        "--out-json", str(tmp_path / "out.json"),
        "--causal-arms", "bogus_arm",
        "--n-samples", "1",
    )
    assert res.returncode == 2, (res.stdout, res.stderr)
    assert "not in" in res.stderr


def test_wrong_placement_must_differ(tmp_path: Path) -> None:
    sft = _make_sft(tmp_path / "sft")
    grpo_av = _make_grpo_av(tmp_path / "grpo_av")
    pairs = tmp_path / "pairs.jsonl"
    _write_pairs(pairs, [_row("goal__traj000000_step000000")])
    res = _run(
        "--sft-dir", str(sft),
        "--grpo-av-dir", str(grpo_av),
        "--pairs-path", str(pairs),
        "--activations-root", str(tmp_path / "acts"),
        "--out-json", str(tmp_path / "out.json"),
        "--causal-arms", "semantic,wrong_placement",
        "--sim-placement", "last_text",
        "--wrong-placement", "last_text",
        "--n-samples", "1",
    )
    assert res.returncode == 2, (res.stdout, res.stderr)
    assert "must differ" in res.stderr


def test_language_swap_is_default(tmp_path: Path) -> None:
    """``--eval-protocol`` defaults to ``language_swap`` (eval-v2)."""
    sft = _make_sft(tmp_path / "sft")
    grpo_av = _make_grpo_av(tmp_path / "grpo_av")
    pairs = tmp_path / "pairs.jsonl"
    _write_pairs(pairs, [_row("goal__traj000000_step000000")])
    res = _run(
        "--sft-dir", str(sft),
        "--grpo-av-dir", str(grpo_av),
        "--pairs-path", str(pairs),
        "--activations-root", str(tmp_path / "acts"),
        "--out-json", str(tmp_path / "out.json"),
        "--n-samples", "1",
        "--dry-run",
        expect_rc=0,
    )
    assert "eval_protocol=language_swap" in res.stdout, res.stdout


def test_legacy_eval_protocol_opt_in(tmp_path: Path) -> None:
    """``--eval-protocol legacy`` is still selectable for regression sweeps."""
    sft = _make_sft(tmp_path / "sft")
    grpo_av = _make_grpo_av(tmp_path / "grpo_av")
    pairs = tmp_path / "pairs.jsonl"
    _write_pairs(pairs, [_row("goal__traj000000_step000000")])
    res = _run(
        "--sft-dir", str(sft),
        "--grpo-av-dir", str(grpo_av),
        "--pairs-path", str(pairs),
        "--activations-root", str(tmp_path / "acts"),
        "--out-json", str(tmp_path / "out.json"),
        "--eval-protocol", "legacy",
        "--n-samples", "1",
        "--dry-run",
        expect_rc=0,
    )
    assert "eval_protocol=legacy" in res.stdout, res.stdout


def test_no_steer_arm_accepted(tmp_path: Path) -> None:
    """``--causal-arms ...,no_steer`` is accepted (new arm)."""
    sft = _make_sft(tmp_path / "sft")
    grpo_av = _make_grpo_av(tmp_path / "grpo_av")
    pairs = tmp_path / "pairs.jsonl"
    _write_pairs(pairs, [_row("goal__traj000000_step000000")])
    res = _run(
        "--sft-dir", str(sft),
        "--grpo-av-dir", str(grpo_av),
        "--pairs-path", str(pairs),
        "--activations-root", str(tmp_path / "acts"),
        "--out-json", str(tmp_path / "out.json"),
        "--causal-arms", "semantic,no_steer",
        "--n-samples", "1",
        "--dry-run",
        expect_rc=0,
    )
    assert "no_steer" in res.stdout, res.stdout


def test_reuse_sft_from_flag_parses(tmp_path: Path) -> None:
    """--reuse-sft-from is accepted; dry-run exits before cache is loaded."""
    sft = _make_sft(tmp_path / "sft")
    grpo_av = _make_grpo_av(tmp_path / "grpo_av")
    pairs = tmp_path / "pairs.jsonl"
    _write_pairs(pairs, [_row(f"goal__traj{i:06d}_step{i:06d}") for i in range(3)])
    sft_cache = tmp_path / "sft_cache.json"
    sft_cache.write_text(json.dumps({
        "version": 1,
        "sim_config": {
            "sim_max_steps": 100,
            "sim_placement": "image_patch",
            "sim_blend": 1.0,
        },
        "samples": [],
    }))
    res = _run(
        "--sft-dir", str(sft),
        "--grpo-av-dir", str(grpo_av),
        "--pairs-path", str(pairs),
        "--activations-root", str(tmp_path / "acts"),
        "--out-json", str(tmp_path / "out.json"),
        "--reuse-sft-from", str(sft_cache),
        "--write-sft-cache", str(tmp_path / "new_cache.json"),
        "--n-samples", "3",
        "--dry-run",
        expect_rc=0,
    )
    assert "CF steer compare: 3 samples" in res.stdout
