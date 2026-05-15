#!/usr/bin/env python
"""Run warm-start labeling on an activation dump.

Example::

    PYTHONPATH=src python scripts/labeling/run_label.py \\
        --activations-root data/activations/droid_smoke \\
        --dataset-root     third_party/Isaac-GR00T/demo_data/droid_sample \\
        --labels-dir       data/labels/droid_smoke \\
        --max-examples 4 --concurrency 4
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--activations-root", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--labels-dir", required=True)
    p.add_argument("--model", default=os.environ.get("OPENAI_LABELING_MODEL", "gpt-5.1-mini"))
    p.add_argument("--tokenizer-repo", default="Qwen/Qwen3-VL-2B-Instruct",
                   help="Defaults to the public Qwen3-VL-2B-Instruct (Cosmos-Reason2-2B "
                        "is gated; tokenizers are identical for text).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--state-name", default=None,
                   help="Human-readable label for the state vector if you wire it in.")
    p.add_argument("--max-examples", type=int, default=None,
                   help="Cap on number of (example, position) pairs to label.")
    p.add_argument("--positions-per-example", type=int, default=1,
                   help="How many distinct positions to sample per activation example. "
                        "1 = the original behavior; 4-8 is a good sweet spot for SFT.")
    p.add_argument("--guarantee-strata", action="store_true",
                   help="When --positions-per-example >= 2, always reserve one slot "
                        "for last_text and one for anchor (when present). Avoids the "
                        "image_patch-dominated ~75/16/8 mix produced by pure "
                        "POSITION_MIX draws against image-heavy sequences. See "
                        "docs/sft_plan/01_data_audit.md.")
    p.add_argument("--no-resume", action="store_true",
                   help="Disable JSONL resume (re-label everything).")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    # Note: we deliberately do NOT redirect HF_HOME. The user's HF auth token
    # is stored under the default ~/.cache/huggingface/, and the tokenizer's
    # config download needs that token (Cosmos-Reason2-2B is gated).
    from nla.labeling.pipeline import run_labeling_sync

    n_new = run_labeling_sync(
        activations_root=args.activations_root,
        dataset_root=args.dataset_root,
        labels_dir=args.labels_dir,
        model=args.model,
        tokenizer_repo=args.tokenizer_repo,
        seed=args.seed,
        concurrency=args.concurrency,
        state_name=args.state_name,
        max_examples=args.max_examples,
        positions_per_example=args.positions_per_example,
        guarantee_strata=args.guarantee_strata,
        resume=not args.no_resume,
    )
    logging.info("Labeled %d new examples.", n_new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
