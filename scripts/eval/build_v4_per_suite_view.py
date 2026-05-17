#!/usr/bin/env python
"""Build per-suite directory layout views for the V4 multimodal judge run.

The existing ``scripts/eval/verify_libero_label_quality.py`` expects a
``--labels-root`` dir containing one ``libero_<suite>/{labels.jsonl,frames_cache/}``
subdir per suite. The V4 dataset ships as either:

 (a) a single combined ``data/labels/libero_4suite_v4_combined/labels.jsonl``
     (with ``meta.suite`` populated and ``meta.source_example_id`` carrying a
     ``{suite}__`` prefix); or
 (b) per-suite ``data/labels/libero_4suite_v4/libero_<suite>/labels.jsonl``
     (no ``meta.suite``, plain ``source_example_id`` matching V3 cache filenames),
     but without any ``frames_cache`` (V4 re-used the V3 cache).

This helper materialises the layout the judge script expects, symlinking the
V3 ``frames_cache/`` directory (the physical frames are identical because the
underlying LIBERO trajectories were not re-rendered).

Run it for each mode::

    PYTHONPATH=src .venv/bin/python scripts/eval/build_v4_per_suite_view.py \
        --mode combined \
        --combined-jsonl data/labels/libero_4suite_v4_combined/labels.jsonl \
        --out-root data/labels/libero_4suite_v4_combined_per_suite \
        --v3-root data/labels/libero_4suite_stride2

    PYTHONPATH=src .venv/bin/python scripts/eval/build_v4_per_suite_view.py \
        --mode per_suite \
        --per-suite-root data/labels/libero_4suite_v4 \
        --out-root data/labels/libero_4suite_v4_view \
        --v3-root data/labels/libero_4suite_stride2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path


SUITES = ("spatial", "goal", "object", "10")


def _ensure_symlink(target: Path, link_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(target.resolve())


def _build_combined(combined_jsonl: Path, out_root: Path, v3_root: Path) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    files = {}
    for s in SUITES:
        sd = out_root / f"libero_{s}"
        sd.mkdir(parents=True, exist_ok=True)
        files[s] = (sd / "labels.jsonl").open("w")
    counts: Counter[str] = Counter()
    stripped: Counter[str] = Counter()
    bad_prefix = 0
    with combined_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            meta = obj.get("meta") or {}
            suite = meta.get("suite")
            if suite not in files:
                continue
            sid = meta.get("source_example_id") or ""
            prefix = f"{suite}__"
            if sid.startswith(prefix):
                meta["source_example_id"] = sid[len(prefix):]
                stripped[suite] += 1
            else:
                bad_prefix += 1
            obj["meta"] = meta
            files[suite].write(json.dumps(obj) + "\n")
            counts[suite] += 1
    for fh in files.values():
        fh.close()
    for s in SUITES:
        v3_cache = v3_root / f"libero_{s}" / "frames_cache"
        link = out_root / f"libero_{s}" / "frames_cache"
        if not v3_cache.exists():
            print(f"WARN: V3 cache missing for {s}: {v3_cache}", file=sys.stderr)
            continue
        _ensure_symlink(v3_cache, link)
    return {
        "per_suite_counts": dict(counts),
        "prefix_stripped": dict(stripped),
        "bad_prefix_rows": bad_prefix,
        "out_root": str(out_root),
    }


def _build_per_suite_view(per_suite_root: Path, out_root: Path,
                          v3_root: Path) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for s in SUITES:
        src_labels = per_suite_root / f"libero_{s}" / "labels.jsonl"
        if not src_labels.exists():
            print(f"WARN: missing labels for {s}: {src_labels}", file=sys.stderr)
            continue
        sd = out_root / f"libero_{s}"
        sd.mkdir(parents=True, exist_ok=True)
        link_labels = sd / "labels.jsonl"
        _ensure_symlink(src_labels, link_labels)
        v3_cache = v3_root / f"libero_{s}" / "frames_cache"
        if not v3_cache.exists():
            print(f"WARN: V3 cache missing for {s}: {v3_cache}", file=sys.stderr)
            continue
        _ensure_symlink(v3_cache, sd / "frames_cache")
        n = 0
        with src_labels.open() as f:
            for _ in f:
                n += 1
        counts[s] = n
    return {"per_suite_counts": counts, "out_root": str(out_root)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", choices=("combined", "per_suite"), required=True)
    p.add_argument("--combined-jsonl",
                   default="data/labels/libero_4suite_v4_combined/labels.jsonl")
    p.add_argument("--per-suite-root",
                   default="data/labels/libero_4suite_v4")
    p.add_argument("--out-root", required=True)
    p.add_argument("--v3-root",
                   default="data/labels/libero_4suite_stride2",
                   help="V3 labels root containing libero_<suite>/frames_cache/.")
    args = p.parse_args(argv)

    out_root = Path(args.out_root)
    v3_root = Path(args.v3_root)
    if args.mode == "combined":
        result = _build_combined(Path(args.combined_jsonl), out_root, v3_root)
    else:
        result = _build_per_suite_view(Path(args.per_suite_root), out_root, v3_root)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
