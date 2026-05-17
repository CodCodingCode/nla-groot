#!/usr/bin/env python
"""Deduplicate labels.jsonl by canonical (source_example_id, position_index, position_type).

Resume-friendly labeling can append duplicate keys if files were merged, parallel runs
raced, or example_id formatting drifted. SFT then sees contradictory targets for the
same activation slice.

Example::

  python scripts/labeling/dedupe_labels.py \\
      --in data/labels/libero_goal_pilot/labels.jsonl \\
      --out data/labels/libero_goal_pilot/labels.dedup.jsonl \\
      --prefer last

  # Replace original (writes .bak first)
  python scripts/labeling/dedupe_labels.py \\
      --in data/labels/libero_goal_pilot/labels.jsonl \\
      --prefer last --in-place
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


def _position_key(obj: dict[str, Any]) -> tuple[str, int, str] | None:
    if obj.get("kind") != "position":
        return None
    m = obj.get("meta") or {}
    sid = m.get("source_example_id")
    pidx = m.get("position_index")
    pt = m.get("position_type")
    if sid is None or pidx is None or pt is None:
        return None
    return (str(sid), int(pidx), str(pt))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inp", required=True, type=Path)
    p.add_argument("--out", dest="out", type=Path)
    p.add_argument(
        "--prefer", choices=("last", "first"), default="last",
        help="which row to keep when duplicate keys disagree",
    )
    p.add_argument(
        "--in-place", action="store_true",
        help="replace --in after copying to .bak (still pass --in)",
    )
    args = p.parse_args()
    inp: Path = args.inp
    if not inp.is_file():
        raise SystemExit(f"not found: {inp}")

    rows: list[dict[str, Any]] = []
    with inp.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    by_key_first: dict[tuple[str, int, str], dict[str, Any]] = {}
    by_key_last: dict[tuple[str, int, str], dict[str, Any]] = {}
    texts_per_key: defaultdict[tuple[str, int, str], set[str]] = defaultdict(set)

    for obj in rows:
        key = _position_key(obj)
        if key is None:
            continue
        text = (obj.get("description") or "").strip()
        if text:
            texts_per_key[key].add(text)
        if key not in by_key_first:
            by_key_first[key] = obj
        by_key_last[key] = obj

    pick = by_key_last if args.prefer == "last" else by_key_first
    n_conflict_keys = sum(1 for k, s in texts_per_key.items() if len(s) > 1)

    out: list[dict[str, Any]] = []
    emitted: set[tuple[str, int, str]] = set()
    for obj in rows:
        key = _position_key(obj)
        if key is None:
            out.append(obj)
            continue
        if key in emitted:
            continue
        out.append(pick[key])
        emitted.add(key)

    if args.in_place:
        bak = inp.with_suffix(inp.suffix + ".bak")
        shutil.copy2(inp, bak)
        out_path = inp
    else:
        if args.out is None:
            raise SystemExit("pass --out or use --in-place")
        out_path = args.out

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for obj in out:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    dropped = len(rows) - len(out)
    print(
        f"rows {len(rows)} -> {len(out)} (dropped {dropped}); "
        f"position keys with conflicting text: {n_conflict_keys}"
    )
    if args.in_place:
        print(f"backup: {bak}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
