#!/usr/bin/env python
"""Warm-start SFT for AV + AR.

Example::

    PYTHONPATH=src python scripts/training/run_sft.py \\
        --activations-root data/activations/droid_smoke \\
        --labels-jsonl     data/labels/droid_smoke/labels.jsonl \\
        --output-dir       data/sft/droid_smoke \\
        --batch-size 4 --total-steps 50 --eval-every 10
"""

from __future__ import annotations

import argparse
import logging
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--activations-root", required=True)
    p.add_argument("--labels-jsonl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--alpha", type=float, default=196.15)
    p.add_argument("--ar-layers", type=int, default=10)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--total-steps", type=int, default=1000)
    p.add_argument("--av-weight", type=float, default=1.0)
    p.add_argument("--ar-weight", type=float, default=1.0)
    p.add_argument("--ar-contrastive-weight", type=float, default=0.0,
                   help="If >0, add InfoNCE term to AR's SFT loss: penalize generic "
                        "descriptions AR can decode into ANY batch row.")
    p.add_argument("--use-quality-weights", action="store_true",
                   help="If set, multiply per-batch losses by mean(quality_weight) read "
                        "from labels.jsonl.  No-op until labels carry the field.")
    p.add_argument("--split-by", choices=["episode", "row"], default="episode",
                   help="Train/val split granularity.  Default 'episode' (needed for "
                        "memorization-vs-generalization measurement); use 'row' only "
                        "as a legacy ablation.")
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--held-out-fraction", type=float, default=0.05)
    p.add_argument("--max-train-items", type=int, default=None)
    p.add_argument("--max-val-items", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    from nla.models import ARConfig, AVConfig
    from nla.training.sft import SFTConfig, run_sft

    av_cfg = AVConfig(
        base_model=args.base_model,
        activation_dim=2048,
        alpha=args.alpha,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        dtype=args.dtype,
    )
    ar_cfg = ARConfig(
        base_model=args.base_model,
        activation_dim=2048,
        alpha=args.alpha,
        truncate_to_n_layers=args.ar_layers,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        dtype=args.dtype,
    )
    cfg = SFTConfig(
        activations_root=args.activations_root,
        labels_jsonl=args.labels_jsonl,
        output_dir=args.output_dir,
        av_cfg=av_cfg,
        ar_cfg=ar_cfg,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        av_weight=args.av_weight,
        ar_weight=args.ar_weight,
        ar_contrastive_weight=args.ar_contrastive_weight,
        use_quality_weights=args.use_quality_weights,
        split_by=args.split_by,
        eval_every=args.eval_every,
        save_every=args.save_every,
        log_every=args.log_every,
        held_out_fraction=args.held_out_fraction,
        max_train_items=args.max_train_items,
        max_val_items=args.max_val_items,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )
    summary = run_sft(cfg)
    logging.info("SFT done. %s", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
