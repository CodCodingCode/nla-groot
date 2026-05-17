#!/usr/bin/env python
"""Mine (scene, target_intent) pairs for sim-success GRPO on LIBERO Goal.

For each labeled position in the Goal split of the combined corpus we emit
a small bundle::

    {
      "example_id":         <label example_id>,
      "source_example_id":  <activation example_id>,
      "episode_index":      <int>,
      "step_index":         <int>,
      "position_type":      "last_text" | "image_patch" | "anchor",
      "position_index":     <int>,
      "source_intent":      <free-text demo task>,
      "source_task":        <canonical task id>,
      "target_intent":      <free-text task we want the policy to do>,
      "target_task":        <canonical task id>,
      "target_env_name":    "libero_sim/<target_task>",
      "is_counterfactual":  bool   (source_task != target_task)
    }

Roughly 50% of rows are "matching" (target == source intent — these test
whether the AV can preserve behavior under steering) and 50% are
"counterfactual" (target differs — the ones the GRPO loss actually needs
to learn to redirect on).

The GRPO trainer reads this file via
:class:`nla.training.counterfactual_data.CounterfactualPairSampler` to draw
``(activation, intent_text)`` pairs each step. The env name lets the
sim worker spin up the appropriate BDDL — which we choose to be the
*target* task so :code:`info["success"]` would fire if the policy actually
executed the steered intent (a useful, free secondary signal even though
our custom predicates run regardless of the loaded BDDL).
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

from nla.eval.steerability.predicates import GOAL_TASKS, resolve_task  # noqa: E402


logger = logging.getLogger("mine_grpo_counterfactuals")


def _resolve_canonical(instruction: str) -> str | None:
    """Free-text instruction -> canonical task id. Returns None if unknown."""
    try:
        return resolve_task(instruction).name
    except KeyError:
        return None


def _iter_goal_label_rows(labels_path: Path) -> Iterable[dict]:
    """Yield only rows from the ``goal`` suite that carry full metadata."""
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
            if meta.get("suite") != "goal":
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
            canon = _resolve_canonical(instr)
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
            }
            n_kept += 1
    logger.info(
        "Goal label scan: total=%d kept=%d  skipped(suite!=goal)=%d  skipped(missing meta)=%d  skipped(unknown intent)=%d",
        n_total, n_kept, n_skipped_suite, n_skipped_meta, n_skipped_intent,
    )


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


def mine_pairs(
    labels_path: Path,
    *,
    out_path: Path,
    seed: int = 0,
    matching_fraction: float = 0.5,
    max_per_episode: int | None = None,
    max_total: int | None = None,
) -> dict:
    """Walk Goal labels and emit one (source, target) per row.

    ``matching_fraction`` of rows get ``target_intent == source_intent`` (the
    "preserve behavior" half of the curriculum); the rest get a uniformly-
    sampled counterfactual intent drawn from the other 9 Goal tasks.

    Pass ``max_per_episode`` to keep the file balanced across episodes when
    the corpus has highly imbalanced row counts per task. Pass ``max_total``
    to cap the output for smoke runs.
    """
    rng = random.Random(seed)
    canon_to_instr = _canonical_to_instruction()
    all_canon = list(GOAL_TASKS.keys())

    if not labels_path.exists():
        raise FileNotFoundError(f"labels.jsonl not found: {labels_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_per_episode: dict[int, int] = {}
    src_task_counts: Counter[str] = Counter()
    tgt_task_counts: Counter[str] = Counter()
    counterfactual_count = 0
    written = 0

    with out_path.open("w") as fout:
        for row in _iter_goal_label_rows(labels_path):
            ep = row["episode_index"]
            if max_per_episode is not None:
                cur = rows_per_episode.get(ep, 0)
                if cur >= max_per_episode:
                    continue
                rows_per_episode[ep] = cur + 1
            src = row["source_task"]
            src_task_counts[src] += 1
            if rng.random() < matching_fraction:
                tgt = src
                is_cf = False
            else:
                tgt = rng.choice([t for t in all_canon if t != src])
                is_cf = True
            tgt_task_counts[tgt] += 1
            if is_cf:
                counterfactual_count += 1
            out_row = {
                **row,
                "target_intent":    canon_to_instr[tgt],
                "target_task":      tgt,
                "target_env_name":  f"libero_sim/{tgt}",
                "is_counterfactual": bool(is_cf),
            }
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            written += 1
            if max_total is not None and written >= max_total:
                break

    summary = {
        "labels_path":      str(labels_path),
        "out_path":         str(out_path),
        "seed":             int(seed),
        "n_rows":           int(written),
        "n_counterfactual": int(counterfactual_count),
        "counterfactual_fraction": float(counterfactual_count / max(1, written)),
        "source_task_counts": dict(src_task_counts),
        "target_task_counts": dict(tgt_task_counts),
    }
    summary_path = out_path.with_suffix(out_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(
        "Wrote %d pairs to %s (counterfactual=%.1f%%); summary at %s",
        written, out_path, 100.0 * summary["counterfactual_fraction"], summary_path,
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--labels", default="data/labels/libero_4suite_combined/labels.jsonl",
        help="Combined labels.jsonl path.",
    )
    p.add_argument(
        "--out", default="data/grpo/libero_goal_counterfactual_pairs.jsonl",
        help="Output JSONL path.",
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
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    mine_pairs(
        labels_path=Path(args.labels),
        out_path=Path(args.out),
        seed=args.seed,
        matching_fraction=args.matching_fraction,
        max_per_episode=args.max_per_episode,
        max_total=args.max_total,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
