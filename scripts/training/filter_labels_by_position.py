#!/usr/bin/env python
"""Filter a labels.jsonl (and its companion hard-negatives JSONL) by position type.

Used to build a "no anchor" parallel corpus for the anchor ablation
(see ``docs/sft_plan/anchor_ablation.md``). Inputs are never mutated.

The labels file is a JSONL where each row has::

    {
      "example_id": "<suite>__traj<...>_step<...>@p<NNN>_<position_type>",
      "description": "- scene: ...",
      "meta": {"position_type": "last_text" | "image_patch" | "anchor", ...},
      ...
    }

The hard-negatives index (produced by ``mine_hard_negatives.py``) is a JSONL
where each row keys ``anchor`` to a label-style ``example_id`` and lists
``negs`` of the same shape. Rows are filtered by the *anchor* row's
``position_type`` (mined negs already share the anchor's ptype by construction,
see ``data/activations/libero_4suite_v4_combined/hard_negatives_v4_audit.md``),
so dropping anchor anchors also removes every neg row whose anchor was an
anchor-ptype label.

Example::

    PYTHONPATH=src python scripts/training/filter_labels_by_position.py \\
        --labels-in        data/labels/libero_4suite_v4_combined/labels.jsonl \\
        --labels-out       data/labels/libero_4suite_v4_combined/labels_no_anchor.jsonl \\
        --hard-negatives-in  data/activations/libero_4suite_v4_combined/hard_negatives.jsonl \\
        --hard-negatives-out data/activations/libero_4suite_v4_combined/hard_negatives_no_anchor.jsonl \\
        --exclude anchor \\
        --audit-out        data/labels/libero_4suite_v4_combined/labels_no_anchor.audit.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path


logger = logging.getLogger("filter_labels_by_position")


def _position_type_of_label(row: dict) -> str | None:
    """Return ``meta.position_type`` if present, else parse from example_id suffix."""
    meta = row.get("meta") or {}
    ptype = meta.get("position_type")
    if ptype:
        return str(ptype)
    ex = row.get("example_id") or ""
    if "@" in ex and "_" in ex:
        return ex.rsplit("_", 1)[-1]
    return None


def _position_type_of_hard_neg(row: dict) -> str | None:
    """Return ``position_type`` if present, else parse from anchor suffix."""
    ptype = row.get("position_type")
    if ptype:
        return str(ptype)
    anchor = row.get("anchor") or ""
    if "@" in anchor and "_" in anchor:
        return anchor.rsplit("_", 1)[-1]
    return None


def filter_labels(
    in_path: Path,
    out_path: Path,
    *,
    exclude: set[str],
) -> dict:
    """Stream-filter labels.jsonl; return stats."""
    if in_path.resolve() == out_path.resolve():
        raise ValueError(f"--labels-in and --labels-out must differ ({in_path})")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    before = Counter()
    after = Counter()
    suite_after = Counter()
    excluded_sample: list[str] = []
    n_in = 0
    n_kept = 0
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            n_in += 1
            row = json.loads(line)
            ptype = _position_type_of_label(row)
            if ptype is None:
                logger.warning(
                    "Row %d: missing position_type; keeping (example_id=%r)",
                    n_in, row.get("example_id"),
                )
            before[ptype or "<unknown>"] += 1
            if ptype in exclude:
                if len(excluded_sample) < 8:
                    excluded_sample.append(row.get("example_id") or "<no id>")
                continue
            fout.write(line + "\n")
            n_kept += 1
            after[ptype or "<unknown>"] += 1
            suite = (row.get("meta") or {}).get("suite") or "<unknown>"
            suite_after[suite] += 1

    return {
        "labels_in_path": str(in_path),
        "labels_out_path": str(out_path),
        "n_in": n_in,
        "n_kept": n_kept,
        "n_dropped": n_in - n_kept,
        "position_type_counts_before": dict(before),
        "position_type_counts_after": dict(after),
        "suite_counts_after": dict(suite_after),
        "excluded_sample_example_ids": excluded_sample,
    }


def filter_hard_negatives(
    in_path: Path,
    out_path: Path,
    *,
    exclude: set[str],
) -> dict:
    """Stream-filter the hard-negatives JSONL keyed on anchor row's position_type."""
    if in_path.resolve() == out_path.resolve():
        raise ValueError(f"--hard-negatives-in and --hard-negatives-out must differ ({in_path})")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    before = Counter()
    after = Counter()
    n_in = 0
    n_kept = 0
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            n_in += 1
            row = json.loads(line)
            ptype = _position_type_of_hard_neg(row)
            before[ptype or "<unknown>"] += 1
            if ptype in exclude:
                continue
            fout.write(line + "\n")
            n_kept += 1
            after[ptype or "<unknown>"] += 1

    return {
        "hard_negatives_in_path": str(in_path),
        "hard_negatives_out_path": str(out_path),
        "n_in": n_in,
        "n_kept": n_kept,
        "n_dropped": n_in - n_kept,
        "anchor_position_type_counts_before": dict(before),
        "anchor_position_type_counts_after": dict(after),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--labels-in", required=True, type=Path,
                   help="Input labels.jsonl.")
    p.add_argument("--labels-out", required=True, type=Path,
                   help="Output filtered labels JSONL (must differ from input).")
    p.add_argument(
        "--exclude", action="append", default=[],
        choices=["last_text", "image_patch", "anchor", "fallback"],
        help="Position type to drop (repeatable). Empty list = no-op copy.",
    )
    p.add_argument(
        "--hard-negatives-in", type=Path, default=None,
        help="Optional companion hard-negatives JSONL "
             "(e.g. data/activations/.../hard_negatives.jsonl).",
    )
    p.add_argument(
        "--hard-negatives-out", type=Path, default=None,
        help="Output filtered hard-negatives JSONL. Required iff --hard-negatives-in is set.",
    )
    p.add_argument(
        "--audit-out", type=Path, default=None,
        help="Optional JSON path for the audit summary. Defaults to "
             "<labels-out>.audit.json.",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    exclude = set(args.exclude or [])
    if not exclude:
        logger.warning("--exclude not set; output will be a verbatim copy.")

    if (args.hard_negatives_in is None) != (args.hard_negatives_out is None):
        raise SystemExit(
            "--hard-negatives-in and --hard-negatives-out must be provided together."
        )

    labels_stats = filter_labels(
        args.labels_in, args.labels_out, exclude=exclude,
    )
    logger.info(
        "labels: kept %d/%d (dropped %d). After-counts: %s",
        labels_stats["n_kept"], labels_stats["n_in"],
        labels_stats["n_dropped"], labels_stats["position_type_counts_after"],
    )

    hard_neg_stats: dict | None = None
    if args.hard_negatives_in is not None:
        hard_neg_stats = filter_hard_negatives(
            args.hard_negatives_in, args.hard_negatives_out, exclude=exclude,
        )
        logger.info(
            "hard_negatives: kept %d/%d (dropped %d).",
            hard_neg_stats["n_kept"], hard_neg_stats["n_in"],
            hard_neg_stats["n_dropped"],
        )
        if hard_neg_stats["n_kept"] != labels_stats["n_kept"]:
            logger.warning(
                "Row count mismatch: labels kept %d, hard_negatives kept %d. "
                "Mining was originally one-row-per-label, so a mismatch usually "
                "means the hard-neg index was built from a different label run.",
                labels_stats["n_kept"], hard_neg_stats["n_kept"],
            )

    audit_path = args.audit_out or args.labels_out.with_suffix(
        args.labels_out.suffix + ".audit.json"
    )
    audit = {
        "exclude": sorted(exclude),
        "labels": labels_stats,
        "hard_negatives": hard_neg_stats,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2))
    logger.info("Wrote audit to %s", audit_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
