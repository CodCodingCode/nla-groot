#!/usr/bin/env python
"""Run one A/B-test round over a list of prompt variants.

Example::

    PYTHONPATH=src python scripts/labeling/run_ab_round.py \
        --round 1 \
        --variants V0,V1,V2,V3,V4,V5,V6 \
        --eval-set data/prompt_ab/eval_set.jsonl \
        --out-root data/prompt_ab \
        --label-model gpt-5.1-mini \
        --grader-model gpt-5.1 \
        --label-concurrency 16 --grade-concurrency 16

Outputs::

    <out-root>/round_NN/variant_<id>/labels.jsonl
    <out-root>/round_NN/variant_<id>/grades.jsonl
    <out-root>/round_NN/claude_samples/<id>.jsonl
    <out-root>/round_NN/scores.json
    <out-root>/round_NN/round.log
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--round", type=int, required=True, help="Round number (1-indexed).")
    p.add_argument("--variants", required=True,
                   help="Comma-separated variant ids registered in prompt_variants, e.g. V0,V1,V2,V3,V4,V5,V6")
    p.add_argument("--eval-set", default="data/prompt_ab/eval_set.jsonl")
    p.add_argument("--out-root", default="data/prompt_ab",
                   help="Round dirs go in <out-root>/round_<NN>/")
    p.add_argument("--label-model", default=os.environ.get("OPENAI_LABELING_MODEL", "gpt-5.1-mini"))
    p.add_argument("--grader-model", default=os.environ.get("OPENAI_GRADER_MODEL", "gpt-5.1"))
    p.add_argument("--label-concurrency", type=int, default=16)
    p.add_argument("--grade-concurrency", type=int, default=16)
    p.add_argument("--pass-threshold", type=float, default=0.95)
    p.add_argument("--skip-distinctness", action="store_true",
                   help="Skip the sentence-transformer (b)-auto scorer (saves ~20s)")
    p.add_argument("--claude-n-per-position", type=int, default=10)
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    round_dir = Path(args.out_root) / f"round_{args.round:02d}"
    round_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging to file + console.
    log_path = round_dir / "round.log"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=handlers,
        force=True,
    )
    logger = logging.getLogger("nla.ab_round")
    logger.info("Round %d, variants=%s, log=%s", args.round, variants, log_path)

    from nla.labeling.ab_test import RoundConfig, run_round_sync

    cfg = RoundConfig(
        round_idx=args.round,
        variants=variants,
        eval_set_path=Path(args.eval_set),
        out_dir=round_dir,
        label_model=args.label_model,
        grader_model=args.grader_model,
        label_concurrency=args.label_concurrency,
        grade_concurrency=args.grade_concurrency,
        skip_distinctness=args.skip_distinctness,
        pass_threshold=args.pass_threshold,
        claude_n_per_position=args.claude_n_per_position,
    )

    scorecards = run_round_sync(cfg)

    # Final summary table
    logger.info("=" * 80)
    logger.info("Round %d summary", args.round)
    logger.info("  variant   axis_a   axis_b   axis_c   pass95?")
    logger.info("  " + "-" * 60)
    winners: list[str] = []
    for variant in variants:
        c = scorecards.get(variant)
        if c is None:
            logger.warning("  %s   <NO SCORE>", variant)
            continue
        logger.info(
            "  %-8s  %.3f    %.3f    %.3f    %s",
            variant, c.pass_rate_a, c.pass_rate_b_combined,
            c.pass_rate_c_combined, "YES" if c.passes_95 else "no",
        )
        if c.passes_95:
            winners.append(variant)
    logger.info("=" * 80)
    if winners:
        logger.info("WINNERS this round (>=%.2f on all 3 axes): %s",
                    args.pass_threshold, winners)
    else:
        logger.info("No variant cleared >=%.2f on all 3 axes this round.",
                    args.pass_threshold)

    return 0 if scorecards else 1


if __name__ == "__main__":
    sys.exit(main())
