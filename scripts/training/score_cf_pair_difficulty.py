#!/usr/bin/env python
"""Annotate CF pair rows with a ``difficulty: float`` field for GRPO curriculum.

The v7 GRPO recipe enables ``--curriculum-easy-to-hard``, which expects each
counterfactual pair JSONL row to carry a numeric ``difficulty`` (0 = easy,
1 = hard). Without it the curriculum flag silently no-ops.

Definition of difficulty for one CF row::

    For the SFT baseline policy (no AR injection), what's the probability
    that running a LIBERO rollout on (target_env_name, target_intent) with
    seed=0 hits the predicate?

    difficulty = 1.0 - mean(predicate_success)  over `--n-seeds` rollouts.

Operationally:
  - Easy pair: SFT baseline solves the target task most of the time.
  - Hard pair: SFT baseline can't solve it even without our steer interfering.

Why this matters for curriculum:
  - Hard pairs early → both arms fail → all rewards zero → no gradient.
  - Easy pairs early → some baseline successes → contrastive arm produces
    real positive/negative reward → policy update has direction.

Usage::

    # Pre-flight: steer server up on --policy-port. We use `--no-steer` arm
    # only, so any AR works (or the SFT base AR).
    PYTHONPATH=src .venv/bin/python scripts/training/score_cf_pair_difficulty.py \\
        --pairs-in  data/grpo/libero_goal_counterfactual_pairs_cfonly.jsonl \\
        --pairs-out data/grpo/libero_goal_counterfactual_pairs_cfonly_difficulty.jsonl \\
        --sft-dir   data/sft/v7_libero_4suite \\
        --activations-root data/activations/libero_4suite_v4_combined \\
        --policy-host localhost --policy-port 5556 \\
        --sim-rollout-python third_party/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python \\
        --n-seeds 2 \\
        --max-rows 200   # subsample for speed; full file is 1193 rows × 2 seeds
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pairs-in", required=True)
    p.add_argument("--pairs-out", required=True)
    p.add_argument("--sft-dir", required=True,
                   help="SFT dir with ar/ (AR is used only as a vector source "
                        "to pass the steer_disabled assemble_jobs path; the "
                        "actual rollout uses options['steer_disabled']=True so "
                        "AR's output never reaches the policy).")
    p.add_argument("--activations-root", required=True)
    p.add_argument("--policy-host", default="localhost")
    p.add_argument("--policy-port", type=int, default=5556)
    p.add_argument("--sim-rollout-python", required=True)
    p.add_argument("--n-seeds", type=int, default=2,
                   help="How many no-steer rollouts per pair. With 2 seeds, "
                        "difficulty resolution is 0/0.5/1.0; with 4 it's 0/0.25/0.5/0.75/1.0.")
    p.add_argument("--sim-max-steps", type=int, default=100)
    p.add_argument("--sim-batch-size", type=int, default=4)
    p.add_argument("--max-rows", type=int, default=None,
                   help="Subsample the pairs file to this many rows (deterministic, "
                        "first N). Useful for previewing difficulty distribution before "
                        "committing to scoring the full file.")
    p.add_argument("--device", default="cuda")
    return p


def _load_pairs(path: Path, max_rows: int | None) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not (r.get("target_task") and r.get("target_env_name")):
                continue
            rows.append(r)
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    pairs_in = Path(args.pairs_in)
    pairs_out = Path(args.pairs_out)
    if pairs_in.resolve() == pairs_out.resolve():
        raise SystemExit("--pairs-in and --pairs-out must differ")

    pairs = _load_pairs(pairs_in, args.max_rows)
    print(f"Loaded {len(pairs)} pairs from {pairs_in}", flush=True)
    if not pairs:
        raise SystemExit("No valid pairs found")

    from nla.extraction.storage import ActivationShardReader
    from nla.training.checkpoint import load_ar_from_sft
    from nla.training.sim_reward import SimRewardWorker, assemble_jobs

    sft_dir = Path(args.sft_dir)
    reader = ActivationShardReader(args.activations_root)
    ar = load_ar_from_sft(sft_dir / "ar", device=args.device, freeze=True)

    # Generate a dummy steer_vec; sim_disabled=True means it won't be used.
    # AR's output dim is what matters; the rollout subprocess will save it as
    # steer_h.npy but the wrapper short-circuits the steer hook.
    from nla.layer_spec import BACKBONE_EMBEDDING_DIM
    dummy_vec = torch.zeros(BACKBONE_EMBEDDING_DIM, dtype=torch.float32)

    worker = SimRewardWorker(
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        n_workers=1,
        sim_batch_size=args.sim_batch_size,
        rollout_python=args.sim_rollout_python,
        cache_path=None,
    )

    pairs_out.parent.mkdir(parents=True, exist_ok=True)

    n_pairs = len(pairs)
    n_total_rollouts = n_pairs * args.n_seeds
    print(f"Scoring {n_pairs} pairs × {args.n_seeds} seeds = {n_total_rollouts} rollouts", flush=True)

    # Build all jobs in one batch — SimRewardWorker handles the parallelism.
    rollout_texts: list[str] = []
    steer_vecs_list: list[torch.Tensor] = []
    target_tasks: list[str] = []
    target_env_names: list[str] = []
    source_ids: list[str] = []
    seeds: list[int] = []
    job_to_pair_idx: list[int] = []

    for i, p in enumerate(pairs):
        for s in range(args.n_seeds):
            rollout_texts.append("")  # unused when steer_disabled=True
            steer_vecs_list.append(dummy_vec)
            target_tasks.append(p["target_task"])
            target_env_names.append(p["target_env_name"])
            source_ids.append(p["source_example_id"])
            seeds.append(s)
            job_to_pair_idx.append(i)

    steer_vecs = torch.stack(steer_vecs_list, dim=0)
    jobs = assemble_jobs(
        rollout_texts=rollout_texts,
        steer_vecs=steer_vecs,
        target_tasks=target_tasks,
        target_env_names=target_env_names,
        source_ids=source_ids,
        seeds=seeds,
        sim_max_steps=args.sim_max_steps,
        placement="image_patch",
        blend=1.0,
        steer_disabled=True,  # no-steer baseline
    )

    print(f"Dispatching {len(jobs)} rollouts to SimRewardWorker (sim_batch_size={args.sim_batch_size})", flush=True)
    t0 = time.time()
    sim_results = worker.compute(jobs)
    print(f"Rollouts done in {(time.time() - t0) / 60:.1f} min", flush=True)

    # Aggregate per pair_idx.
    per_pair_hits: dict[int, list[int]] = {}
    for res, pair_idx in zip(sim_results, job_to_pair_idx):
        hit = 1 if (res.predicate or 0) > 0 else 0
        per_pair_hits.setdefault(pair_idx, []).append(hit)

    # Write annotated pairs.
    n_with_difficulty = 0
    difficulty_distribution: dict[float, int] = {}
    with pairs_out.open("w") as f:
        for i, p in enumerate(pairs):
            hits = per_pair_hits.get(i, [])
            if hits:
                success_rate = float(np.mean(hits))
                difficulty = 1.0 - success_rate
                p["difficulty"] = round(difficulty, 4)
                p["baseline_success_rate"] = round(success_rate, 4)
                p["n_difficulty_seeds"] = len(hits)
                n_with_difficulty += 1
                difficulty_distribution[difficulty] = difficulty_distribution.get(difficulty, 0) + 1
            f.write(json.dumps(p) + "\n")

    print(f"\nWrote {pairs_out} ({n_with_difficulty}/{len(pairs)} annotated)")
    print(f"Difficulty distribution:")
    for d in sorted(difficulty_distribution):
        n = difficulty_distribution[d]
        print(f"  difficulty={d:.2f}  baseline_succ={1-d:.2f}  n={n} ({100*n/len(pairs):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
