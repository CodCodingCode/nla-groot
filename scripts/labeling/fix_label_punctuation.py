#!/usr/bin/env python
"""Append a trailing period to position-row descriptions that lack one.

Reads ``--labels`` (a JSONL produced by the labeling pipeline) and, for every
row with ``kind == "position"`` whose ``description.strip()`` does not end in
``.``, ``!``, or ``?``, appends a single ``.``.  Writes to a sibling ``.tmp``
file then atomically replaces ``--labels``.

This is a tiny cosmetic sweep meant to be run after ``relabel_bad_rows.py``.
A backup at ``--labels`` + ``.bak2`` is assumed to already exist (created by
the merge step in the relabel script); pass ``--backup`` if you want this
script to make its own ``.bak3`` copy first.

Example::

    python scripts/labeling/fix_label_punctuation.py \\
        --labels data/labels/libero_goal_pilot/labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

_END_PUNCT = (".", "!", "?")


def _needs_period(desc: str) -> bool:
    s = (desc or "").rstrip()
    if not s:
        return False
    return s[-1] not in _END_PUNCT


def _add_period(desc: str) -> str:
    """Append '.' but preserve trailing whitespace/newlines after it."""
    rstripped = desc.rstrip()
    trailer = desc[len(rstripped):]
    return rstripped + "." + trailer


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--labels", required=True, type=Path)
    p.add_argument("--backup", action="store_true",
                   help="Copy --labels to <labels>.bak3 before rewriting.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report counts but do not write.")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    labels_path = Path(args.labels)
    if not labels_path.exists():
        raise SystemExit(f"labels file not found: {labels_path}")

    if args.backup and not args.dry_run:
        bak = labels_path.with_suffix(labels_path.suffix + ".bak3")
        shutil.copy2(labels_path, bak)
        logging.info("backup: %s", bak)

    n_total = 0
    n_position = 0
    n_fixed = 0
    n_skipped_empty = 0

    tmp = labels_path.with_suffix(labels_path.suffix + ".tmp")
    out_handle = None if args.dry_run else tmp.open("w")
    try:
        with labels_path.open() as src:
            for raw in src:
                line = raw.rstrip("\n")
                n_total += 1
                if not line:
                    if out_handle:
                        out_handle.write("\n")
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    if out_handle:
                        out_handle.write(line + "\n")
                    continue
                if obj.get("kind") != "position":
                    if out_handle:
                        out_handle.write(line + "\n")
                    continue
                n_position += 1
                desc = obj.get("description") or ""
                if not desc.strip():
                    n_skipped_empty += 1
                    if out_handle:
                        out_handle.write(line + "\n")
                    continue
                if _needs_period(desc):
                    obj["description"] = _add_period(desc)
                    n_fixed += 1
                    if out_handle:
                        out_handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
                else:
                    if out_handle:
                        out_handle.write(line + "\n")
    finally:
        if out_handle:
            out_handle.close()

    logging.info(
        "rows=%d position=%d empty=%d period-appended=%d (%s)",
        n_total, n_position, n_skipped_empty, n_fixed,
        "DRY-RUN" if args.dry_run else "WROTE",
    )

    if not args.dry_run:
        tmp.replace(labels_path)
    elif tmp.exists():
        tmp.unlink()

    return 0


if __name__ == "__main__":
    sys.exit(main())
