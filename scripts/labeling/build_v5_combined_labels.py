#!/usr/bin/env python
"""Merge per-suite V5 expanded labels into one combined labels.jsonl.

Expects each suite at ``<per_suite_root>/libero_<suite>/labels.jsonl`` with
``meta.suite`` and prefixed ``example_id`` (from ``expand_step_labels --suite``).

Writes ``<out>/labels.jsonl`` and ``<out>/_merge_summary.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("build_v5_combined_labels")

SUITES = ("spatial", "goal", "object", "10")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--per-suite-root",
        type=Path,
        default=Path("data/labels/libero_4suite_v5"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/labels/libero_4suite_v5_combined"),
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "labels.jsonl"
    stats: dict[str, int] = {}
    total = 0

    with out_path.open("w") as fout:
        for suite in SUITES:
            src = args.per_suite_root / f"libero_{suite}" / "labels.jsonl"
            if not src.is_file():
                logger.error("missing %s", src)
                return 1
            n = 0
            with src.open() as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    fout.write(line + "\n")
                    n += 1
            stats[suite] = n
            total += n
            logger.info("merged suite=%s rows=%d", suite, n)

    summary = {"suites": stats, "total_rows": total, "label_version": "v5"}
    (args.out / "_merge_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
