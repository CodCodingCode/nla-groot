#!/usr/bin/env python
"""Build the V4 selective re-label queue across the four LIBERO suites.

Path A from ``docs/sft_plan/v4_repair/libero_v4_dataset_repair_e1c2dc05.plan.md``:

* ``libero_spatial`` -- ALL rows are queued (the V3 spatial-grounding
  failure was systemic; SA2's pilot proved a full re-label is needed).
* ``libero_goal`` / ``libero_object`` / ``libero_10`` -- selective. Queue a row
  iff its V3 description matches ANY of the V4 failure heuristics:

    1. Non-canonical bullet header (V4 forbids ``gripper:``, ``motion:``,
       ``image_region:``).
    2. Motor-imperative phrase (``V4_MOTOR_IMPERATIVE_PHRASES``).
    3. Scaffold-leakage phrase (``V4_SCAFFOLD_FORBIDDEN_PHRASES``).
    4. Description has fewer than 3 bullet lines (degenerate).
    5. ``error`` field is non-null (failed V3 grader call).

Outputs:

* ``<out-dir>/libero_<suite>.jsonl`` -- one row per queued example, with
  fields ``{example_id, source_example_id, position_index, position_type,
  reasons, instruction}``.
* ``<out-dir>/_summary.json`` -- per-suite counts + reason histograms +
  estimated cost.

Example::

    PYTHONPATH=src python scripts/labeling/build_v4_relabel_queue.py \\
        --v3-labels-root data/labels/libero_4suite_stride2 \\
        --out-dir data/labels/v4_relabel_queue
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from nla.labeling.prompts import (  # noqa: E402
    V4_FORBIDDEN_HEADERS,
    V4_MOTOR_IMPERATIVE_PHRASES,
    V4_SCAFFOLD_FORBIDDEN_PHRASES,
)


SUITES: tuple[str, ...] = (
    "libero_spatial",
    "libero_goal",
    "libero_object",
    "libero_10",
)

# Suites where we re-label every V3 row regardless of heuristic match.
FULL_RELABEL_SUITES: frozenset[str] = frozenset({"libero_spatial"})


# Cost assumption from the plan.
COST_PER_ROW_USD: float = 0.0007


def _phrase_regex(phrases: tuple[str, ...]) -> re.Pattern[str]:
    if not phrases:
        return re.compile(r"$^")  # never matches
    alts = "|".join(re.escape(p) for p in phrases)
    return re.compile(rf"\b(?:{alts})\b", re.IGNORECASE)


_FORBIDDEN_HEADER_RE = re.compile(
    rf"^\s*-?\s*({'|'.join(re.escape(h) for h in V4_FORBIDDEN_HEADERS)})\s*:",
    re.IGNORECASE | re.MULTILINE,
)
_MOTOR_RE = _phrase_regex(V4_MOTOR_IMPERATIVE_PHRASES)
_SCAFFOLD_RE = _phrase_regex(V4_SCAFFOLD_FORBIDDEN_PHRASES)
_BULLET_LINE_RE = re.compile(r"^\s*-\s*[a-z_]+\s*:", re.IGNORECASE | re.MULTILINE)


def _row_key(obj: dict) -> tuple[str, int, str] | None:
    if obj.get("kind") != "position":
        return None
    m = obj.get("meta") or {}
    sid = m.get("source_example_id")
    pidx = m.get("position_index")
    pt = m.get("position_type")
    if sid is None or pidx is None or pt is None:
        return None
    return (str(sid), int(pidx), str(pt))


def _classify_row(obj: dict) -> list[str]:
    """Return the list of failure-heuristic reasons that match this V3 row.

    Empty list means the row is clean from V4's standpoint and would be
    skipped under the selective policy.
    """
    reasons: list[str] = []
    desc = obj.get("description") or ""
    if obj.get("error"):
        reasons.append("error")
    if _FORBIDDEN_HEADER_RE.search(desc):
        reasons.append("forbidden_header")
    if _MOTOR_RE.search(desc):
        reasons.append("motor_imperative")
    if _SCAFFOLD_RE.search(desc):
        reasons.append("scaffold_leakage")
    n_bullets = len(_BULLET_LINE_RE.findall(desc))
    if n_bullets < 3:
        reasons.append("too_few_bullets")
    return reasons


def _process_suite(
    suite: str,
    labels_path: Path,
    out_path: Path,
    *,
    max_rows: int | None,
) -> dict:
    """Scan one suite's V3 labels.jsonl and write the queue JSONL.

    Returns a per-suite summary dict.
    """
    full = suite in FULL_RELABEL_SUITES
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_queued = 0
    n_skipped_clean = 0
    n_unparseable = 0
    reason_counts: Counter[str] = Counter()
    seen_keys: set[tuple[str, int, str]] = set()

    with labels_path.open() as src, out_path.open("w") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                n_unparseable += 1
                continue
            n_total += 1

            key = _row_key(obj)
            if key is None:
                n_unparseable += 1
                continue
            if key in seen_keys:
                continue  # de-dup defensively
            seen_keys.add(key)

            reasons = _classify_row(obj)
            if full:
                if not reasons:
                    reasons = ["full_relabel"]
            elif not reasons:
                n_skipped_clean += 1
                continue

            for r in reasons:
                reason_counts[r] += 1

            sid, pidx, pt = key
            meta = obj.get("meta") or {}
            row = {
                "example_id": obj.get("example_id"),
                "source_example_id": sid,
                "position_index": pidx,
                "position_type": pt,
                "reasons": reasons,
                "instruction": meta.get("instruction"),
                "episode_index": meta.get("episode_index"),
                "step_index": meta.get("step_index"),
            }
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_queued += 1
            if max_rows is not None and n_queued >= max_rows:
                break

    return {
        "suite": suite,
        "labels_path": str(labels_path),
        "queue_path": str(out_path),
        "policy": "full" if full else "selective",
        "n_v3_rows_scanned": n_total,
        "n_unparseable": n_unparseable,
        "n_queued": n_queued,
        "n_skipped_clean": n_skipped_clean,
        "queued_pct": (100.0 * n_queued / n_total) if n_total else 0.0,
        "reason_counts": dict(reason_counts.most_common()),
        "estimated_cost_usd": round(n_queued * COST_PER_ROW_USD, 4),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--v3-labels-root",
        type=Path,
        default=Path("data/labels/libero_4suite_stride2"),
        help="Root containing one subdir per suite; each has labels.jsonl.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/labels/v4_relabel_queue"),
        help="Per-suite queue JSONL files + _summary.json land here.",
    )
    p.add_argument(
        "--max-per-suite",
        type=int,
        default=None,
        help="Cap on rows per suite (debug / cost rehearsal).",
    )
    p.add_argument(
        "--suites",
        nargs="+",
        default=list(SUITES),
        help="Restrict to a subset of suites.",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    for suite in args.suites:
        suite_dir = args.v3_labels_root / suite
        labels_path = suite_dir / "labels.jsonl"
        if not labels_path.is_file():
            logging.warning("V3 labels missing for suite=%s at %s", suite, labels_path)
            continue
        out_path = args.out_dir / f"{suite}.jsonl"
        summary = _process_suite(
            suite, labels_path, out_path, max_rows=args.max_per_suite,
        )
        summaries.append(summary)
        logging.info(
            "  %s -> queued %d / %d rows (%s, $%.2f); reasons: %s",
            suite,
            summary["n_queued"],
            summary["n_v3_rows_scanned"],
            summary["policy"],
            summary["estimated_cost_usd"],
            summary["reason_counts"],
        )

    total_queued = sum(s["n_queued"] for s in summaries)
    total_cost = round(total_queued * COST_PER_ROW_USD, 4)
    summary_path = args.out_dir / "_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "v3_labels_root": str(args.v3_labels_root),
                "out_dir": str(args.out_dir),
                "max_per_suite": args.max_per_suite,
                "cost_per_row_usd": COST_PER_ROW_USD,
                "full_relabel_suites": sorted(FULL_RELABEL_SUITES),
                "total_queued": total_queued,
                "total_estimated_cost_usd": total_cost,
                "per_suite": summaries,
            },
            indent=2,
        )
    )
    logging.info("Wrote summary to %s", summary_path)

    print()
    print("=== V4 re-label queue ===")
    for s in summaries:
        print(
            f"  {s['suite']:15s}  policy={s['policy']:9s}  "
            f"queued={s['n_queued']:>7,d} / {s['n_v3_rows_scanned']:>7,d}  "
            f"({s['queued_pct']:5.1f}%)  ~${s['estimated_cost_usd']:6.2f}"
        )
    print(f"  {'TOTAL':15s}  {' ':17s} queued={total_queued:>7,d}  "
          f"~${total_cost:6.2f}")
    print(f"\nWill re-label {total_queued} rows ~ ${total_cost:.2f} "
          f"(at ${COST_PER_ROW_USD}/row).")
    if total_cost > 50.0:
        print(
            f"\nWARNING: estimated cost ${total_cost:.2f} > $50 budget. "
            "Consider re-running with --max-per-suite or escalating before "
            "kicking off the full driver run."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
