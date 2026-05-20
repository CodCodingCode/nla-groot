#!/usr/bin/env python
"""Mine (scene, target_intent) pairs for sim-success GRPO on LIBERO suites.

For each labeled position in a chosen suite (``goal``, ``spatial``, ``object``,
``10``) we emit a JSONL row keyed by ``example_id`` / ``source_example_id``.

Sim predicates currently score **Goal** tasks only, so ``target_task`` and
``target_env_name`` always reference a Goal benchmark task whose bodies were
verified in that task's BDDL. Non-goal source rows therefore get cross-suite
Goal sim targets (lookup coverage + valid sim reward).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

# Make `nla` importable when run from the repo root with `PYTHONPATH=src`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from nla.eval.steerability.bddl_bodies import (  # noqa: E402
    DEFAULT_GOAL_BDDL_DIR,
    filter_tasks_with_bodies_in_bddl,
    validate_cf_target_bodies,
)
from nla.eval.steerability.libero_suites import resolve_suite_instruction  # noqa: E402
from nla.eval.steerability.predicates import GOAL_TASKS, resolve_task  # noqa: E402


logger = logging.getLogger("mine_grpo_counterfactuals")

SUITE_CHOICES = ("goal", "spatial", "object", "10", "all")


def _resolve_canonical(instruction: str, suite_label: str = "goal") -> str | None:
    """Free-text instruction -> canonical task id for ``suite_label``."""
    if suite_label == "goal":
        try:
            return resolve_task(instruction).name
        except KeyError:
            return None
    return resolve_suite_instruction(instruction, suite_label)


def _iter_suite_label_rows(labels_path: Path, suite_label: str) -> Iterable[dict]:
    """Yield label rows for one ``meta.suite`` value with full metadata."""
    n_total = 0
    n_kept = 0
    n_skipped_suite = 0
    n_skipped_meta = 0
    n_skipped_intent = 0
    with labels_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("error"):
                continue
            meta = obj.get("meta") or {}
            if meta.get("suite") != suite_label:
                n_skipped_suite += 1
                continue
            sid = meta.get("source_example_id")
            ep = meta.get("episode_index")
            st = meta.get("step_index")
            pos = meta.get("position_index")
            ptype = meta.get("position_type")
            instr = meta.get("instruction")
            if sid is None or ep is None or st is None or pos is None or ptype is None or not instr:
                n_skipped_meta += 1
                continue
            canon = _resolve_canonical(instr, suite_label)
            if canon is None:
                n_skipped_intent += 1
                continue
            yield {
                "example_id":        obj.get("example_id"),
                "source_example_id": str(sid),
                "episode_index":     int(ep),
                "step_index":        int(st),
                "position_index":    int(pos),
                "position_type":     str(ptype),
                "source_intent":     str(instr),
                "source_task":       canon,
                "source_suite":      suite_label,
            }
            n_kept += 1
    logger.info(
        "%s label scan: total=%d kept=%d  skipped(suite!=%s)=%d  "
        "skipped(missing meta)=%d  skipped(unknown intent)=%d",
        suite_label, n_total, n_kept, suite_label,
        n_skipped_suite, n_skipped_meta, n_skipped_intent,
    )


def _iter_goal_label_rows(labels_path: Path) -> Iterable[dict]:
    """Backward-compatible alias."""
    return _iter_suite_label_rows(labels_path, "goal")


def _canonical_to_instruction() -> dict[str, str]:
    """Map canonical task id -> the demo-style instruction string we feed to AV.

    The instructions we use here are the ones present in
    ``data/labels/libero_4suite_combined/labels.jsonl::meta.instruction``,
    so the AV sees the exact same wording it was trained on for matching
    cases. We picked the most-common form from the corpus.
    """
    return {
        "put_the_bowl_on_the_plate":                       "put the bowl on the plate",
        "put_the_bowl_on_the_stove":                       "put the bowl on the stove",
        "put_the_bowl_on_top_of_the_cabinet":              "put the bowl on top of the cabinet",
        "put_the_wine_bottle_on_the_rack":                 "put the wine bottle on the rack",
        "put_the_wine_bottle_on_top_of_the_cabinet":       "put the wine bottle on top of the cabinet",
        "put_the_cream_cheese_in_the_bowl":                "put the cream cheese in the bowl",
        "push_the_plate_to_the_front_of_the_stove":        "push the plate to the front of the stove",
        "open_the_top_drawer_and_put_the_bowl_inside":     "open the top drawer and put the bowl inside",
        "open_the_middle_drawer_of_the_cabinet":           "open the middle drawer of the cabinet",
        "turn_on_the_stove":                               "turn on the stove",
    }


def _weighted_counterfactual_target(
    *,
    rng: random.Random,
    src: str,
    all_canon: list[str],
    tgt_emit: Counter[str],
) -> str:
    """Pick a counterfactual target weighted inverse to current emit count.

    Targets with lower current counts get exponentially more weight. This
    rescues the long-tail tasks when uniform sampling would just track the
    source-task imbalance.
    """
    options = [t for t in all_canon if t != src]
    # Smoothed inverse-count weights; +1 in denom avoids div-by-zero.
    weights = [1.0 / (1.0 + tgt_emit[t]) for t in options]
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for opt, w in zip(options, weights):
        acc += w
        if r <= acc:
            return opt
    return options[-1]


def mine_pairs(
    labels_path: Path,
    *,
    out_path: Path,
    suite_label: str = "goal",
    seed: int = 0,
    matching_fraction: float = 0.5,
    max_per_episode: int | None = None,
    max_total: int | None = None,
    max_per_source_task: int | None = None,
    balance_target_counts: bool = False,
    shuffle: bool = True,
) -> dict:
    """Walk one suite's labels and emit (source, Goal sim target) pairs.

    For ``suite_label == "goal"``, ``matching_fraction`` rows keep
    ``target_task == source_task``. For other suites, targets are always
    drawn from :data:`GOAL_TASKS` (sim predicates require Goal tasks); those
    rows are always marked counterfactual w.r.t. the source suite task.
    """
    if suite_label not in ("goal", "spatial", "object", "10"):
        raise ValueError(f"unknown suite_label {suite_label!r}")

    rng = random.Random(seed)
    goal_canon_to_instr = _canonical_to_instruction()
    all_goal = list(GOAL_TASKS.keys())

    if not labels_path.exists():
        raise FileNotFoundError(f"labels.jsonl not found: {labels_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_per_episode: dict[int, int] = {}
    src_task_counts: Counter[str] = Counter()
    tgt_task_counts: Counter[str] = Counter()
    src_task_emitted: Counter[str] = Counter()
    counterfactual_count = 0
    written = 0
    n_rejected_bodies = 0
    instance_cache: dict[str, frozenset[str]] = {}
    bddl_dir = DEFAULT_GOAL_BDDL_DIR

    row_stream = list(_iter_suite_label_rows(labels_path, suite_label))
    if shuffle:
        rng.shuffle(row_stream)

    with out_path.open("w") as fout:
        for row in row_stream:
            ep = row["episode_index"]
            if max_per_episode is not None:
                cur = rows_per_episode.get(ep, 0)
                if cur >= max_per_episode:
                    continue
            src = row["source_task"]
            if max_per_source_task is not None and src_task_emitted[src] >= max_per_source_task:
                continue
            if max_per_episode is not None:
                rows_per_episode[ep] = rows_per_episode.get(ep, 0) + 1
            src_task_emitted[src] += 1
            src_task_counts[src] += 1

            use_matching = (
                suite_label == "goal"
                and src in GOAL_TASKS
                and rng.random() < matching_fraction
            )
            if use_matching:
                tgt = src
                is_cf = False
            else:
                options = [t for t in all_goal if t != src or suite_label != "goal"]
                valid = filter_tasks_with_bodies_in_bddl(
                    options, bddl_dir, instance_cache=instance_cache,
                )
                if not valid:
                    n_rejected_bodies += 1
                    continue
                if balance_target_counts:
                    tgt = _weighted_counterfactual_target(
                        rng=rng, src=src, all_canon=valid,
                        tgt_emit=tgt_task_counts,
                    )
                else:
                    tgt = rng.choice(valid)
                is_cf = True

            tgt_task_counts[tgt] += 1
            if is_cf:
                counterfactual_count += 1
            out_row = {
                **row,
                "target_intent": goal_canon_to_instr[tgt],
                "target_task": tgt,
                "target_env_name": f"libero_sim/{tgt}",
                "is_counterfactual": bool(is_cf),
            }
            body_issues = validate_cf_target_bodies(
                tgt, out_row["target_env_name"], bddl_dir,
                instance_cache=instance_cache,
            )
            if body_issues:
                n_rejected_bodies += 1
                logger.debug(
                    "[bddl] skip sid=%s tgt=%s: %s",
                    row.get("source_example_id"), tgt, body_issues,
                )
                continue
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            written += 1
            if max_total is not None and written >= max_total:
                break

    summary = {
        "labels_path":      str(labels_path),
        "out_path":         str(out_path),
        "suite_label":      suite_label,
        "seed":             int(seed),
        "n_rows":           int(written),
        "n_counterfactual": int(counterfactual_count),
        "counterfactual_fraction": float(counterfactual_count / max(1, written)),
        "source_task_counts": dict(src_task_counts),
        "target_task_counts": dict(tgt_task_counts),
        "n_rejected_missing_bodies": int(n_rejected_bodies),
    }
    summary_path = out_path.with_suffix(out_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(
        "Wrote %d pairs to %s (counterfactual=%.1f%%, rejected_bodies=%d); summary at %s",
        written, out_path, 100.0 * summary["counterfactual_fraction"],
        n_rejected_bodies, summary_path,
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--labels", default="data/labels/libero_4suite_combined/labels.jsonl",
        help="Combined labels.jsonl path.",
    )
    p.add_argument(
        "--suite",
        default="goal",
        choices=list(SUITE_CHOICES),
        help="Label suite to mine (goal/spatial/object/10) or all four sequentially.",
    )
    p.add_argument(
        "--out", default="data/grpo/libero_goal_counterfactual_pairs.jsonl",
        help="Output JSONL path (for --suite all, used as output directory if it "
             "ends with /, else files are libero_{suite}_counterfactual_pairs.jsonl "
             "alongside this path's parent).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--matching-fraction", type=float, default=0.5,
        help="Fraction of rows where target_intent == source_intent.",
    )
    p.add_argument(
        "--max-per-episode", type=int, default=None,
        help="Optional cap on rows per (episode, suite) for balance.",
    )
    p.add_argument(
        "--max-total", type=int, default=None,
        help="Optional cap on total rows (for smoke runs).",
    )
    p.add_argument(
        "--max-per-source-task", type=int, default=None,
        help="Optional hard cap on rows per canonical source task. Used to "
             "trim the head of an over-represented source task (e.g. when one "
             "task hogs >25%% of the corpus). Default off preserves byte-"
             "identical behavior.",
    )
    p.add_argument(
        "--balance-target-counts", action="store_true",
        help="Replace the uniform counterfactual target sampler with a "
             "weighted one that inverse-weights by current target-task emit "
             "count. Useful when source-task imbalance leaves some Goal tasks "
             "below the audit gate's 5%% floor on the target side. Default "
             "off preserves byte-identical behavior.",
    )
    p.add_argument(
        "--no-shuffle", dest="shuffle", action="store_false", default=True,
        help="Iterate Goal label rows in file order instead of shuffling. "
             "Default is to shuffle (seeded by --seed) so a small --max-total "
             "samples uniformly across the corpus instead of reading the head "
             "of the JSONL (which is grouped by task).",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def _out_path_for_suite(base: Path, suite_label: str) -> Path:
    if base.suffix == ".jsonl":
        parent = base.parent
        if suite_label == "goal":
            return base
        return parent / f"libero_{suite_label}_counterfactual_pairs.jsonl"
    return base / f"libero_{suite_label}_counterfactual_pairs.jsonl"


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    labels_path = Path(args.labels)
    out_base = Path(args.out)
    suites = (
        ["goal", "spatial", "object", "10"]
        if args.suite == "all"
        else [args.suite]
    )
    for i, suite in enumerate(suites):
        out_path = _out_path_for_suite(out_base, suite)
        mine_pairs(
            labels_path=labels_path,
            out_path=out_path,
            suite_label=suite,
            seed=int(args.seed) + i,
            matching_fraction=args.matching_fraction,
            max_per_episode=args.max_per_episode,
            max_total=args.max_total,
            max_per_source_task=args.max_per_source_task,
            balance_target_counts=bool(args.balance_target_counts),
            shuffle=bool(args.shuffle),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
