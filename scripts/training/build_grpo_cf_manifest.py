#!/usr/bin/env python
"""Build a GRPO train-pool manifest from mined counterfactual pairs JSONLs.

The manifest lists every activation ``example_id`` that has at least one CF
row (via ``source_example_id``). Pass it to ``run_grpo.py`` as
``--cf-eligible-ids-path`` so ``SampledPositionDataset`` only draws
activations that will get sim reward — no wasted steps on missing pairs.

Example::

    PYTHONPATH=src python scripts/training/build_grpo_cf_manifest.py \\
        --pairs data/grpo/libero_goal_counterfactual_pairs.jsonl \\
        --pairs-extra data/grpo/libero_spatial_counterfactual_pairs.jsonl \\
        --pairs-extra data/grpo/libero_object_counterfactual_pairs.jsonl \\
        --pairs-extra data/grpo/libero_10_counterfactual_pairs.jsonl \\
        --out data/grpo/libero_4suite_cf_eligible.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from nla.training.counterfactual_data import (  # noqa: E402
    MANIFEST_VERSION,
    collect_cf_eligible_example_ids,
)

logger = logging.getLogger("build_grpo_cf_manifest")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--pairs", required=True,
        help="Primary CF pairs JSONL (same as --sim-counterfactual-pairs-path).",
    )
    p.add_argument(
        "--pairs-extra", default=[], action="append",
        help="Extra CF pairs JSONL(s). Repeat for each file merged at train time.",
    )
    p.add_argument(
        "--out", required=True,
        help="Output manifest JSON path.",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s [%(name)s] %(message)s",
    )

    paths = [Path(args.pairs), *[Path(p) for p in args.pairs_extra]]
    ids = collect_cf_eligible_example_ids(paths)
    sorted_ids = sorted(ids)
    suite_counts = Counter(eid.split("__", 1)[0] for eid in sorted_ids)

    manifest = {
        "version": MANIFEST_VERSION,
        "description": (
            "Activation example_ids eligible for sim-GRPO "
            "(each has >=1 mined counterfactual pair)."
        ),
        "pairs_paths": [str(p) for p in paths],
        "n_example_ids": len(sorted_ids),
        "example_ids_by_suite": dict(sorted(suite_counts.items())),
        "example_ids": sorted_ids,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))
    logger.info(
        "Wrote %d example_ids to %s (suites: %s)",
        len(sorted_ids), out_path, dict(suite_counts),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
