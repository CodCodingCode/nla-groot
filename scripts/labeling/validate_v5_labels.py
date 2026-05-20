#!/usr/bin/env python
"""Validate V5 nested step labels in labels_steps.jsonl.

Each row may store the nested object at the top level, under ``slots``, or as
JSON inside ``description``.

Usage::

    PYTHONPATH=src python scripts/labeling/validate_v5_labels.py \\
        --in data/labels/my_run/labels_steps.jsonl

    PYTHONPATH=src python scripts/labeling/validate_v5_labels.py \\
        --in data/labels/my_run/labels_steps.jsonl --show-errors 5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from nla.labeling.schema_v5 import (
    cross_slot_jaccard,
    extract_nested_from_row,
    validate_nested,
)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                rows.append({"_parse_error": str(e), "_line": line_no})
    return rows


def _row_id(row: dict, line_no: int) -> str:
    for key in ("example_id", "source_example_id", "step_id"):
        if row.get(key):
            return str(row[key])
    return f"line_{line_no}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--in", dest="inp", required=True, type=Path)
    p.add_argument(
        "--show-errors",
        type=int,
        default=10,
        help="Max invalid rows to print (0 = summary only)",
    )
    p.add_argument(
        "--jaccard",
        action="store_true",
        help="Print mean cross-slot Jaccard for valid rows (scene/target)",
    )
    args = p.parse_args(argv)

    if not args.inp.is_file():
        print(f"not found: {args.inp}", file=sys.stderr)
        return 1

    rows = _load_jsonl(args.inp)
    n_ok = 0
    n_bad = 0
    n_skip = 0
    err_counter: Counter[str] = Counter()
    scene_jac: list[float] = []
    target_jac: list[float] = []
    bad_examples: list[tuple[str, list[str]]] = []

    for i, row in enumerate(rows, 1):
        if "_parse_error" in row:
            n_bad += 1
            err_counter["json_parse"] += 1
            if len(bad_examples) < args.show_errors:
                bad_examples.append((f"line_{row.get('_line', i)}", [row["_parse_error"]]))
            continue

        nested = extract_nested_from_row(row)
        if nested is None:
            n_skip += 1
            continue

        ok, errors, norm = validate_nested(nested)
        if ok:
            n_ok += 1
            if args.jaccard:
                jac = cross_slot_jaccard(norm)
                scene_jac.append(jac["scene"])
                target_jac.append(jac["target"])
        else:
            n_bad += 1
            for e in errors:
                err_counter[e.split(":")[0] if ":" in e else e] += 1
            if args.show_errors and len(bad_examples) < args.show_errors:
                bad_examples.append((_row_id(row, i), errors))

    print(f"file: {args.inp}")
    print(f"rows: {len(rows)}  valid: {n_ok}  invalid: {n_bad}  no_nested: {n_skip}")
    if err_counter:
        print("error kinds (prefix):")
        for k, v in err_counter.most_common(20):
            print(f"  {k}: {v}")
    if args.jaccard and scene_jac:
        print(
            f"mean cross-slot jaccard: scene={sum(scene_jac)/len(scene_jac):.3f} "
            f"target={sum(target_jac)/len(target_jac):.3f} (n={len(scene_jac)})"
        )
    for rid, errors in bad_examples:
        print(f"\n--- {rid} ---")
        for e in errors[:12]:
            print(f"  {e}")

    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
