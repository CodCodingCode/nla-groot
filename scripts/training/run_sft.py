#!/usr/bin/env python
"""Warm-start SFT for AV + AR.

Example (LIBERO smoke)::

    PYTHONPATH=src python scripts/training/run_sft.py \\
        --activations-root data/activations/libero_goal_pilot \\
        --labels-jsonl     data/labels/libero_goal_pilot/labels.jsonl \\
        --stats-json       data/activations/libero_goal_pilot/stats.json \\
        --output-dir       data/sft/libero_goal_pilot_smoke \\
        --batch-size 4 --total-steps 50 --eval-every 10

The script is corpus-agnostic: substitute any extraction root + labels file
produced by ``scripts/extraction/run_extract.py`` and
``scripts/labeling/run_label.py``. Always pass ``--stats-json`` so AV/AR see
the corpus-specific alpha (P75 norm); a mismatched alpha silently miscalibrates
MSE and the closed-loop FVE.
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
    p.add_argument("--alpha", type=float, default=197.44,
                   help="Activation L2-norm scaling factor (P75 ‖h‖₂). Overridden by "
                        "--stats-json when that is provided.")
    p.add_argument("--stats-json", default=None,
                   help="Path to a Phase-1 extraction stats.json. When set, α is read "
                        "from its p75_norm and overrides --alpha for both AV and AR.")
    p.add_argument("--ar-layers", type=int, default=16,
                   help="Number of decoder layers AR keeps after truncation. Default 16 "
                        "matches GR00T's SELECT_LAYER so AR depth mirrors the activation's "
                        "training layer.")
    p.add_argument("--ar-clip-target-scaled", type=float, default=None,
                   help="If set, clamp the α-scaled AR target to ±value during "
                        "forward_sft (e.g. 5.0). Tames heavy tails; no effect on inference.")
    p.add_argument("--ar-nce-temperature", type=float, default=0.1,
                   help="Temperature for AR's InfoNCE contrastive sims. Lower = sharper "
                        "softmax = stronger contrast. 0.1 is a standard contrastive "
                        "default and works at batch 4.")
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
    p.add_argument(
        "--ar-nce-hard-negative-source",
        choices=["none", "same_episode", "same_position_type", "topk_cosine"],
        default="none",
        help="Hard-negative mining for AR's InfoNCE term. 'none' (default) keeps "
             "the legacy in-batch-only contrast. 'same_episode' samples K negatives "
             "per anchor from the SAME episode but a DIFFERENT step (visually-similar "
             "scene, different timestep). 'same_position_type' samples from a "
             "DIFFERENT episode whose label has the same position_type. 'topk_cosine' "
             "loads a precomputed JSONL of activation-cosine top-K neighbors (see "
             "scripts/training/mine_hard_negatives.py); requires "
             "--ar-nce-hard-negative-index-path. Requires --ar-contrastive-weight > 0 "
             "to take effect.",
    )
    p.add_argument(
        "--ar-nce-hard-negatives-per-anchor",
        type=int,
        default=4,
        help="K_neg: number of hard-negative captions sampled per anchor when "
             "--ar-nce-hard-negative-source != none.",
    )
    p.add_argument(
        "--ar-nce-hard-negative-index-path",
        default=None,
        help="Path to the mining JSONL produced by mine_hard_negatives.py. Required "
             "when --ar-nce-hard-negative-source=topk_cosine. Ignored otherwise.",
    )
    p.add_argument("--use-quality-weights", action="store_true",
                   help="If set, multiply per-batch losses by mean(quality_weight) read "
                        "from labels.jsonl.  No-op until labels carry the field.")
    p.add_argument("--split-by", choices=["episode", "row"], default="episode",
                   help="Train/val split granularity.  Default 'episode' (needed for "
                        "memorization-vs-generalization measurement); use 'row' only "
                        "as a legacy ablation.")
    p.add_argument("--balance-position-mix", action="store_true",
                   help="Draw training rows with a WeightedRandomSampler so per-batch "
                        "position_type frequencies approximate layer_spec.POSITION_MIX "
                        "(40/40/20). Use when the labels file is skewed.")
    p.add_argument(
        "--image-patch-pooling",
        choices=["pinned", "mean_pool_image", "strided_image", "center_image"],
        default="pinned",
        help="V4 image-patch pooling. 'pinned' (default) preserves V3 behaviour "
             "(use the single labeled position_index per row). The non-pinned "
             "options pool over every valid image-patch token at read time and "
             "are applied ONLY to rows whose position_type == 'image_patch' "
             "(last_text / anchor rows are untouched). Per "
             "data/sft/libero_4suite_v3/v4_extraction_scorecard.json the "
             "recommended V4 setting is 'mean_pool_image'.",
    )
    p.add_argument(
        "--image-patch-pooling-strided-k",
        type=int,
        default=4,
        help="K for --image-patch-pooling=strided_image (number of evenly-spaced "
             "image-patch tokens to average). Ignored for other pooling modes.",
    )
    p.add_argument("--min-bullets", type=int, default=None,
                   help="Drop labels whose description has fewer than this many '-' "
                        "bullet lines. Use to filter degenerate captions.")
    p.add_argument("--eval-closed-loop", action="store_true",
                   help="Run an extra closed-loop validation pass per eval: "
                        "h -> AV.generate -> AR -> ĥ, stratified by position_type. "
                        "Slow; consider combining with --closed-loop-max-batches.")
    p.add_argument("--closed-loop-temps", type=float, nargs="+", default=(0.0,),
                   help="Sampling temperatures for closed-loop AV generation. "
                        "0.0 == greedy (do_sample=False).")
    p.add_argument("--closed-loop-max-batches", type=int, default=None,
                   help="Cap the number of val batches used per closed-loop eval pass.")
    p.add_argument(
        "--ar-av-mix-max",
        type=float,
        default=0.3,
        help="Scheduled sampling for AR only: after warmup, per-batch probability ramps "
             "linearly up to this value (0 disables). AV CE always uses gold captions. "
             "V3 default 0.3 (was 0.0); pass 0 to revert to gold-only.",
    )
    p.add_argument(
        "--ar-av-mix-warmup-frac",
        type=float,
        default=0.5,
        help="Fraction of total_steps with p=0 before ramping to --ar-av-mix-max.",
    )
    p.add_argument(
        "--ar-av-mix-max-new-tokens",
        type=int,
        default=96,
        help="Max new tokens for AV.generate when feeding AR under mixing.",
    )
    p.add_argument(
        "--ar-av-mix-sample",
        action="store_true",
        help="Sample from AV during mixing (temperature 0.7); default is greedy.",
    )
    p.add_argument(
        "--ar-av-mix-log-text-every",
        type=int,
        default=200,
        help="Log one truncated AV-mix caption every N steps when mixing fires (0=off).",
    )
    p.add_argument("--no-episode-split-fallback", action="store_true",
                   help="When --split-by=episode but the dump has <2 distinct "
                        "episode_index values, fail with RuntimeError instead of "
                        "silently falling back to a row split. Use for paper / "
                        "generalization runs where the val split must be honest.")
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

    alpha = args.alpha
    if args.stats_json:
        from nla.extraction.stats import load_stats
        stats = load_stats(args.stats_json)
        alpha = float(stats.p75_norm)
        logging.info(
            "Loaded α=%.6f from %s (p75_norm; overrides --alpha=%.6f).",
            alpha, args.stats_json, args.alpha,
        )

    av_cfg = AVConfig(
        base_model=args.base_model,
        activation_dim=2048,
        alpha=alpha,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        dtype=args.dtype,
    )
    ar_cfg = ARConfig(
        base_model=args.base_model,
        activation_dim=2048,
        alpha=alpha,
        truncate_to_n_layers=args.ar_layers,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        dtype=args.dtype,
        clip_target_scaled=args.ar_clip_target_scaled,
        nce_temperature=args.ar_nce_temperature,
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
        ar_nce_hard_negative_source=args.ar_nce_hard_negative_source,
        ar_nce_hard_negatives_per_anchor=args.ar_nce_hard_negatives_per_anchor,
        ar_nce_hard_negative_index_path=args.ar_nce_hard_negative_index_path,
        use_quality_weights=args.use_quality_weights,
        split_by=args.split_by,
        allow_episode_split_row_fallback=not args.no_episode_split_fallback,
        eval_every=args.eval_every,
        save_every=args.save_every,
        log_every=args.log_every,
        held_out_fraction=args.held_out_fraction,
        max_train_items=args.max_train_items,
        max_val_items=args.max_val_items,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        balance_position_mix=args.balance_position_mix,
        min_bullet_lines=args.min_bullets,
        eval_closed_loop=args.eval_closed_loop,
        closed_loop_temperatures=tuple(args.closed_loop_temps),
        closed_loop_max_batches=args.closed_loop_max_batches,
        ar_av_mix_max=args.ar_av_mix_max,
        ar_av_mix_warmup_frac=args.ar_av_mix_warmup_frac,
        ar_av_mix_max_new_tokens=args.ar_av_mix_max_new_tokens,
        ar_av_mix_do_sample=args.ar_av_mix_sample,
        ar_av_mix_log_text_every=args.ar_av_mix_log_text_every,
        image_patch_pooling=args.image_patch_pooling,
        image_patch_pooling_strided_k=args.image_patch_pooling_strided_k,
    )
    summary = run_sft(cfg)
    logging.info("SFT done. %s", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
