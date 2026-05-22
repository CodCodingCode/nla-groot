#!/usr/bin/env python
"""Build a frozen held-out CF eval slice for sim-GRPO steerability evaluation.

Why this exists
---------------

``build_grpo_cf_manifest.py`` builds a **train pool** manifest (every CF-eligible
``example_id``). It does not separate train from held-out activations. Reporting
"GRPO beat SFT on predicate" against the same pool we trained on is a leakage
trap and the V1 pilot's predicate rate is not a publishable headline.

This script builds the **eval-side** complement: a frozen pair JSONL of CF rows
whose ``source_example_id`` belongs to the held-out episode split that
``run_grpo.py`` constructs via ``SampledPositionDataset(held_out=True, ...)``.

It uses the *same* episode-stratified split function
(:func:`nla.training.dataset._split_episode_aware`) with the *same* ``seed``,
``held_out_fraction``, and ``split_by`` defaults, so the eval slice is exactly
the GRPO val split.

Outputs:

* ``<out>_pairs.jsonl``         — held-out CF rows only (drop-in
  ``--pairs-path`` for ``compare_cf_steer_checkpoints.py``).
* ``<out>_eval_manifest.json``  — metadata + held-out ``example_ids`` (for
  ``--exclude-ids-path`` / ``--require-held-out`` guards).
* ``<out>_train_manifest.json`` — complementary train ``example_ids`` (for
  leakage assertions).

Example::

    PYTHONPATH=src python scripts/training/build_grpo_cf_eval_manifest.py \\
        --pairs data/grpo/libero_goal_counterfactual_pairs.jsonl \\
        --pairs-extra data/grpo/libero_spatial_counterfactual_pairs.jsonl \\
        --pairs-extra data/grpo/libero_object_counterfactual_pairs.jsonl \\
        --pairs-extra data/grpo/libero_10_counterfactual_pairs.jsonl \\
        --activations-root data/activations/libero_4suite_v4_combined \\
        --seed 0 --held-out-fraction 0.05 --split-by episode \\
        --slice all \\
        --out data/grpo/libero_4suite_cf_eval
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from nla.extraction.storage import ActivationShardReader  # noqa: E402
from nla.training.counterfactual_data import MANIFEST_VERSION  # noqa: E402
from nla.training.dataset import _split_episode_aware  # noqa: E402

logger = logging.getLogger("build_grpo_cf_eval_manifest")


_SLICES = ("all", "goal_only", "matched_bddl", "counterfactual_only")


def _slice_keep(row: dict, slice_name: str) -> bool:
    """Return True if a pair row passes the requested slice filter."""
    if slice_name == "all":
        return True
    if slice_name == "goal_only":
        sid = str(row.get("source_example_id", ""))
        return sid.startswith("goal__")
    if slice_name == "matched_bddl":
        return (
            str(row.get("source_task", "")) == str(row.get("target_task", ""))
            and not bool(row.get("is_counterfactual", False))
        )
    if slice_name == "counterfactual_only":
        return bool(row.get("is_counterfactual", False))
    raise ValueError(f"unknown slice: {slice_name!r}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--pairs", required=True,
        help="Primary CF pairs JSONL (matches GRPO --sim-counterfactual-pairs-path).",
    )
    p.add_argument(
        "--pairs-extra", default=[], action="append",
        help="Extra CF pairs JSONL(s); repeat for each file merged at train time.",
    )
    p.add_argument(
        "--activations-root", required=True,
        help="Activation extraction root; used to read ExampleRecord.episode_index "
             "for the episode-stratified split.",
    )
    p.add_argument(
        "--out", required=True,
        help="Output prefix. Produces <out>_pairs.jsonl, <out>_eval_manifest.json, "
             "<out>_train_manifest.json.",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Must match run_grpo.py --seed (default 0).",
    )
    p.add_argument(
        "--held-out-fraction", type=float, default=0.05,
        help="Must match run_grpo.py --held-out-fraction (default 0.05).",
    )
    p.add_argument(
        "--split-by", choices=("episode", "row"), default="episode",
        help="Must match run_grpo.py --split-by (default episode).",
    )
    p.add_argument(
        "--allow-row-fallback", action="store_true",
        help="Permit fallback to row-level split when episode_index is missing. "
             "Off by default; recommended OFF for publishable eval.",
    )
    p.add_argument(
        "--slice", default="all", choices=_SLICES,
        help="Optional eval slice filter (default: all).",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def _iter_pairs(paths: Iterable[Path]):
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"CF pairs file not found: {path}")
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield path, row


def main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s [%(name)s] %(message)s",
    )

    paths = [Path(args.pairs), *[Path(p) for p in args.pairs_extra]]

    reader = ActivationShardReader(args.activations_root)
    n_total = len(reader)
    logger.info(
        "Loaded %d ExampleRecord(s) from %s", n_total, args.activations_root,
    )

    all_indices = list(range(n_total))
    held_indices = _split_episode_aware(
        all_indices,
        reader.records,
        seed=args.seed,
        held_out_fraction=args.held_out_fraction,
        held_out=True,
        split_by=args.split_by,
        label_for_logs="build_grpo_cf_eval_manifest(held_out)",
        allow_row_fallback=args.allow_row_fallback,
    )
    train_indices = _split_episode_aware(
        all_indices,
        reader.records,
        seed=args.seed,
        held_out_fraction=args.held_out_fraction,
        held_out=False,
        split_by=args.split_by,
        label_for_logs="build_grpo_cf_eval_manifest(train)",
        allow_row_fallback=args.allow_row_fallback,
    )
    held_ids = {reader.records[i].example_id for i in held_indices}
    train_ids = {reader.records[i].example_id for i in train_indices}
    overlap = held_ids & train_ids
    if overlap:
        raise RuntimeError(
            f"train/held-out splits overlap on {len(overlap)} example_ids; "
            "this is a bug in _split_episode_aware wiring."
        )
    logger.info(
        "split: %d held-out / %d train example_ids (held-out fraction %.3f)",
        len(held_ids), len(train_ids),
        len(held_ids) / max(1, len(held_ids) + len(train_ids)),
    )

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_pairs_path = out_prefix.with_name(out_prefix.name + "_pairs.jsonl")
    out_eval_manifest = out_prefix.with_name(out_prefix.name + "_eval_manifest.json")
    out_train_manifest = out_prefix.with_name(out_prefix.name + "_train_manifest.json")

    n_in = 0
    n_kept_held = 0
    n_dropped_train = 0
    n_dropped_unknown = 0
    n_dropped_slice = 0
    kept_ids: set[str] = set()
    kept_suites: Counter = Counter()
    kept_target_tasks: Counter = Counter()
    with out_pairs_path.open("w") as out_f:
        for _path, row in _iter_pairs(paths):
            n_in += 1
            sid = str(row.get("source_example_id") or "")
            if not sid:
                continue
            if sid in train_ids:
                n_dropped_train += 1
                continue
            if sid not in held_ids:
                n_dropped_unknown += 1
                continue
            if not _slice_keep(row, args.slice):
                n_dropped_slice += 1
                continue
            n_kept_held += 1
            kept_ids.add(sid)
            kept_suites[sid.split("__", 1)[0]] += 1
            kept_target_tasks[str(row.get("target_task", ""))] += 1
            out_f.write(json.dumps(row) + "\n")

    logger.info(
        "scanned %d rows: kept=%d held-out (slice=%s), "
        "dropped_train=%d, dropped_unknown_id=%d, dropped_slice=%d",
        n_in, n_kept_held, args.slice,
        n_dropped_train, n_dropped_unknown, n_dropped_slice,
    )
    logger.info(
        "kept activations: %d unique source_example_ids across suites %s",
        len(kept_ids), dict(kept_suites),
    )

    common = {
        "version": MANIFEST_VERSION,
        "pairs_paths": [str(p) for p in paths],
        "activations_root": str(args.activations_root),
        "seed": args.seed,
        "held_out_fraction": args.held_out_fraction,
        "split_by": args.split_by,
        "slice": args.slice,
    }
    sorted_held = sorted(held_ids)
    sorted_train = sorted(train_ids)
    eval_manifest = {
        **common,
        "description": (
            "Held-out activation example_ids for sim-GRPO steer eval "
            "(episode-stratified, same split as run_grpo.py val set)."
        ),
        "kind": "eval",
        "n_example_ids": len(sorted_held),
        "example_ids_by_suite": dict(sorted(
            Counter(eid.split("__", 1)[0] for eid in sorted_held).items()
        )),
        "n_kept_pair_rows": n_kept_held,
        "n_kept_pair_source_ids": len(kept_ids),
        "kept_target_tasks": dict(sorted(kept_target_tasks.items())),
        "example_ids": sorted_held,
        "eval_pairs_path": str(out_pairs_path),
    }
    train_manifest = {
        **common,
        "description": (
            "Train-side activation example_ids (complement of the held-out "
            "eval split). Use as --exclude-ids-path leakage guard."
        ),
        "kind": "train",
        "n_example_ids": len(sorted_train),
        "example_ids_by_suite": dict(sorted(
            Counter(eid.split("__", 1)[0] for eid in sorted_train).items()
        )),
        "example_ids": sorted_train,
    }

    out_eval_manifest.write_text(json.dumps(eval_manifest, indent=2))
    out_train_manifest.write_text(json.dumps(train_manifest, indent=2))
    logger.info("Wrote eval pairs:     %s", out_pairs_path)
    logger.info("Wrote eval manifest:  %s", out_eval_manifest)
    logger.info("Wrote train manifest: %s", out_train_manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
