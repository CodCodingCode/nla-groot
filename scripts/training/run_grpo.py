#!/usr/bin/env python
"""GRPO RL fine-tuning for the AV (Verbalizer).

Loads a warm-start SFT checkpoint, freezes it as the reference policy and as
the reward model (AR), and trains the policy AV with::

    reward(h, y) = -|| (h / alpha) - AR(y) ||^2

KL anchor to the frozen reference at coefficient ``--beta``.

Example::

    PYTHONPATH=src python scripts/training/run_grpo.py \\
        --sft-dir          data/sft/droid_ep1_v1 \\
        --activations-root data/activations/droid_ep1 \\
        --output-dir       data/grpo/droid_ep1_b002 \\
        --beta 0.02 --total-steps 200 --rollouts-per-activation 4
"""

from __future__ import annotations

import argparse
import logging
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sft-dir", required=True,
                   help="Output dir from run_sft.py (must contain av/ and ar/).")
    p.add_argument("--activations-root", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--batch-size", type=int, default=4,
                   help="Distinct activations per step (B).")
    p.add_argument("--rollouts-per-activation", type=int, default=4,
                   help="K: rollouts per activation, the GRPO group size.")
    p.add_argument("--rollout-max-new-tokens", type=int, default=160)
    p.add_argument("--rollout-temperature", type=float, default=1.0)
    p.add_argument("--rollout-top-p", type=float, default=0.95)

    p.add_argument("--beta", type=float, default=0.02,
                   help="KL coefficient (paper sweep: {0.01, 0.02, 0.05}).")
    p.add_argument("--no-advantage-normalize", action="store_true")
    p.add_argument("--advantage-clip", type=float, default=None)
    p.add_argument("--ar-co-train-weight", type=float, default=0.0,
                   help="If >0, unfreeze AR and add ar_weight * MSE(AR(rollouts), h/alpha) to the loss.")

    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--learning-rate", type=float, default=3e-6)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--total-steps", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.0)

    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--eval-max-examples", type=int, default=64)

    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--held-out-fraction", type=float, default=0.05)
    p.add_argument("--split-by", choices=["episode", "row"], default="episode",
                   help="Train/val split granularity.  Default 'episode' (needed for "
                        "memorization-vs-generalization measurement); use 'row' only "
                        "as a legacy ablation.")
    p.add_argument("--no-episode-split-fallback", action="store_true",
                   help="When --split-by=episode but the dump has <2 distinct "
                        "episode_index values, fail with RuntimeError instead of "
                        "silently falling back to a row split. Use for paper / "
                        "generalization runs where the val split must be honest.")
    p.add_argument("--eval-temperatures", default="0.0,0.7,1.0",
                   help="Comma-separated rollout temperatures for evaluation.  The "
                        "gap between greedy (0.0) and sampled FVE is itself a "
                        "memorization diagnostic.")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    from nla.training.grpo import GRPOConfig, run_grpo

    eval_temps = tuple(
        float(t.strip()) for t in args.eval_temperatures.split(",") if t.strip()
    )

    cfg = GRPOConfig(
        sft_dir=args.sft_dir,
        activations_root=args.activations_root,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        rollouts_per_activation=args.rollouts_per_activation,
        rollout_max_new_tokens=args.rollout_max_new_tokens,
        rollout_temperature=args.rollout_temperature,
        rollout_top_p=args.rollout_top_p,
        beta=args.beta,
        advantage_normalize=not args.no_advantage_normalize,
        advantage_clip=args.advantage_clip,
        ar_co_train_weight=args.ar_co_train_weight,
        grad_accum_steps=args.grad_accum_steps,
        grad_clip=args.grad_clip,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
        save_every=args.save_every,
        log_every=args.log_every,
        eval_max_examples=args.eval_max_examples,
        gradient_checkpointing=args.gradient_checkpointing,
        held_out_fraction=args.held_out_fraction,
        split_by=args.split_by,
        allow_episode_split_row_fallback=not args.no_episode_split_fallback,
        eval_temperatures=eval_temps,
    )
    summary = run_grpo(cfg)
    logging.info("GRPO done. %s", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
