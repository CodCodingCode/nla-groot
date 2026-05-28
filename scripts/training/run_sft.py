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
import json
import logging
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--recipe",
        choices=["v7"],
        default=None,
        help="Apply a named training recipe before parsing other CLI flags. "
             "Recipe defaults are overridden by any explicit CLI argument. "
             "'v7' is the full retrain plan: policy-effect loss primary, "
             "spatial AR head, training-time blend=0.5, batch-stratified "
             "positions, scheduled-sampling ramped to 0.7. See "
             "docs/sft_plan/v7_runbook.md for the per-setting rationale and "
             "the list of external paths the recipe still needs from you.",
    )
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
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader worker processes. 0 = main-thread loading "
                        "(legacy default). 4-8 typically lifts GPU util from "
                        "~37%% to 80%%+ when action_consistency is enabled. "
                        "Each worker forks the dataset (incl. hard-neg cache "
                        "+ label index), so memory grows linearly with this.")
    p.add_argument("--no-pin-memory", dest="pin_memory", action="store_false",
                   help="Disable pin_memory in DataLoaders. Default is on when "
                        "num_workers > 0. Ignored when num_workers == 0.")
    p.set_defaults(pin_memory=True)
    p.add_argument("--no-persistent-workers", dest="persistent_workers",
                   action="store_false",
                   help="Disable persistent_workers in DataLoaders. Default "
                        "is on when num_workers > 0. Ignored when num_workers == 0.")
    p.set_defaults(persistent_workers=True)
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
    p.add_argument("--position-mix-json", default=None,
                   help="Optional JSON object of target position-type weights for "
                        "the rebalancing sampler (e.g. "
                        '\'{"last_text": 0.5, "image_patch": 0.5}\' for the '
                        "no-anchor ablation). Only consulted when "
                        "--balance-position-mix is set. Omit to use "
                        "layer_spec.POSITION_MIX (40/40/20).")
    p.add_argument(
        "--image-patch-pooling",
        choices=[
            "pinned", "mean_pool_image", "strided_image",
            "strided_image_multi", "center_image",
        ],
        default="pinned",
        help="V4/V5 image-patch pooling. 'pinned' (default) preserves V3 "
             "behaviour (use the single labeled position_index per row). The "
             "single-vector options ('mean_pool_image', 'strided_image', "
             "'center_image') pool over every valid image-patch token at read "
             "time and feed AV a single [H] vector. V5 'strided_image_multi' "
             "gives AV the full [K, H] strided patch grid (one slot per patch) "
             "while AR still regresses against a single [H] (mean over K). "
             "Pooling is applied ONLY to rows whose position_type == "
             "'image_patch' (last_text / anchor rows are untouched). Per "
             "data/sft/libero_4suite_v3/v4_extraction_scorecard.json the "
             "recommended V4 setting was 'mean_pool_image'; V5 default is "
             "'strided_image_multi' with K=8.",
    )
    p.add_argument(
        "--image-patch-pooling-strided-k",
        type=int,
        default=4,
        help="K for --image-patch-pooling=strided_image / strided_image_multi "
             "(number of evenly-spaced image-patch tokens to read). For the "
             "single-vector 'strided_image' the K patches are averaged; for "
             "the multi-vector 'strided_image_multi' they are returned as K "
             "separate AV slot vectors. Ignored for other pooling modes.",
    )
    p.add_argument(
        "--exclude-position-types",
        nargs="+",
        default=None,
        help="Drop rows whose position_type is in this list before splitting "
             "(safety net for the V5 no-anchor arm). Canonical V5 workflow is "
             "to point --labels-jsonl at the pre-filtered labels_no_anchor.jsonl "
             "instead; use this flag when running on the full combined file. "
             "Example: --exclude-position-types anchor.",
    )
    p.add_argument(
        "--include-position-types",
        nargs="+",
        default=None,
        help="Stage-2 plan: positive include filter. Keep only rows whose "
             "position_type is in this list. Use 'image_patch' alone for the "
             "image_patch-only ablation that isolates whether the codec can "
             "learn vision-grounded structure when not diluted with text/"
             "anchor rows. Example: --include-position-types image_patch. "
             "Use --balance-position-mix --position-mix-json instead to "
             "oversample image_patch while keeping all three roles.",
    )
    p.add_argument(
        "--ar-head-type",
        choices=["scalar", "spatial"],
        default="scalar",
        help="Stage-3 plan: AR output shape. 'scalar' (default) returns one "
             "(B, H) vector per text, broadcast across image_patch slots at "
             "inject time. 'spatial' returns (B, N, H) — one vector per "
             "image_patch position — so the live policy sees spatially-varied "
             "vision representations like real GR00T input. Pair with "
             "--ar-spatial-n-positions equal to the K used at extraction "
             "(K=8 with strided_image_multi, matches GR00T's strided patch "
             "count).",
    )
    p.add_argument(
        "--ar-spatial-n-positions",
        type=int,
        default=0,
        help="Number of spatial positions the AR spatial head emits. Required "
             "and must be > 0 when --ar-head-type=spatial. Set to the K used "
             "at extraction (default 8 with strided_image_multi pooling).",
    )
    p.add_argument(
        "--ar-spatial-head-hidden",
        type=int,
        default=0,
        help="Hidden width of the spatial head's intermediate MLP. 0 (default) "
             "uses the AR base model's hidden size — adds no extra hyperparam "
             "knob beyond head_type and n_positions.",
    )
    p.add_argument(
        "--av-prompt-version",
        choices=["legacy", "context_v5"],
        default="context_v5",
        help="V5 default: render the context-enriched AV prompt with Position "
             "type, Timestep, Task instruction, then the activation slot(s). "
             "Pass 'legacy' to fall back to the V3/V4 two-line prompt "
             "byte-identical to the original (useful for ablations or for "
             "continuing a V4 checkpoint without a prompt shift).",
    )
    p.add_argument(
        "--av-no-step-index",
        action="store_true",
        help="Skip the Timestep line in the AV prompt even when "
             "--av-prompt-version=context_v5. The labeler never showed the AV "
             "the literal step index, so this lets you A/B whether the timestep "
             "improves grounding or just adds prompt tokens.",
    )
    p.add_argument(
        "--av-no-instruction",
        action="store_true",
        help="Skip the Task instruction line in the AV prompt even when "
             "--av-prompt-version=context_v5. Useful for measuring how much "
             "of V5's lift comes from the instruction vs the patch fan-out.",
    )
    p.add_argument(
        "--ar-prompt-version",
        choices=["legacy", "context_v5"],
        default="legacy",
        help="AR prompt template. 'legacy' keeps the canonical Summary-only "
             "line (V3/V4 byte-identical). 'context_v5' prepends position type, "
             "timestep, and task instruction before the Summary line.",
    )
    p.add_argument(
        "--ar-no-step-index",
        action="store_true",
        help="Skip the Timestep line in the AR prompt when "
             "--ar-prompt-version=context_v5.",
    )
    p.add_argument(
        "--ar-no-instruction",
        action="store_true",
        help="Skip the Task instruction line in the AR prompt when "
             "--ar-prompt-version=context_v5.",
    )
    p.add_argument(
        "--av-num-image-slots",
        type=int,
        default=8,
        help="K for the multi-slot image_patch prompt (V5 default 8). Each "
             "slot maps to one entry of the K-patch activation tensor. K=1 "
             "falls back to the single-slot prompt regardless of pooling. "
             "Only consulted on image_patch rows; last_text / anchor / "
             "fallback rows always use one slot.",
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
    p.add_argument(
        "--action-consistency-weight",
        type=float,
        default=0.0,
        help="If >0, add an action-head consistency loss (||π(h_real) - π(AR(z))||²) "
             "to the SFT objective. Requires --action-consistency-policy-path and "
             "--action-consistency-dataset-roots. Default 0 keeps SFT untouched.",
    )
    p.add_argument(
        "--action-consistency-every-n-steps",
        type=int,
        default=8,
        help="Cadence for the consistency forward (1 = every step). The policy "
             "forward dominates per-step cost so the default skips most steps.",
    )
    p.add_argument(
        "--action-consistency-max-microbatch",
        type=int,
        default=1,
        help="Number of batch rows fed into the consistency forward per active step. "
             "Defaults to 1 because the frozen GR00T forward is VRAM-heavy.",
    )
    p.add_argument(
        "--action-consistency-image-patch-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set (default), only feed rows whose position_type == 'image_patch' "
             "into the consistency forward, matching the steering placement. "
             "Pass --no-action-consistency-image-patch-only to include all "
             "position types (slower, currently has no eval-time counterpart).",
    )
    p.add_argument(
        "--action-consistency-blend",
        type=float,
        default=1.0,
        help="Steering hook blend factor used inside the action-consistency "
             "forward (SteerSpec.blend). 1.0 = hard replace (legacy). 0.5 = "
             "mix 50/50 with the live activation. The v7 recipe sets 0.5 so "
             "the training-time steer matches what eval/sim use, making the "
             "blend=0.5 dose in-distribution (Stage 0 showed both alpha=1.0 "
             "and alpha=0.5 were OOD with the legacy hard-replace training).",
    )
    p.add_argument(
        "--batch-stratified-positions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When set together with --balance-position-mix, enforce "
             "per-batch position-type quotas (largest-remainder method) "
             "rather than relying on WeightedRandomSampler's epoch-level "
             "rebalance. Ensures every batch contains a guaranteed number "
             "of image_patch rows, which the v7 policy-effect SFT needs to "
             "produce a steady image_patch gradient.",
    )
    p.add_argument(
        "--action-consistency-policy-path",
        default=None,
        help="Path to a frozen GR00T checkpoint used for the consistency forward. "
             "Required when --action-consistency-weight > 0.",
    )
    p.add_argument(
        "--action-consistency-embodiment-tag",
        default=None,
        help="GR00T embodiment tag for the policy loader (e.g. LIBERO_PANDA). "
             "Required when --action-consistency-weight > 0.",
    )
    p.add_argument(
        "--action-consistency-dataset-roots",
        default=None,
        help="JSON mapping {\"suite\": \"<lerobot_dataset_root>\"} used to replay the "
             "original observation per labeled row. Use \"\" as the suite key for "
             "single-suite dumps. Required when --action-consistency-weight > 0.",
    )
    p.add_argument(
        "--action-consistency-manifest-cache",
        default=None,
        help="Where to cache the replay manifest JSONL. Defaults to "
             "<output_dir>/aux/replay_manifest.jsonl.",
    )
    p.add_argument(
        "--action-consistency-suites",
        nargs="+",
        default=None,
        help="Optional whitelist of suite names; only labeled rows whose "
             "suite is in this list participate in the consistency forward. "
             "All other suites still flow through the regular SFT losses. "
             "Default: no filter (use every suite in --action-consistency-dataset-roots).",
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
    parser = _build_parser()
    # Apply named-recipe defaults *before* parse_args so explicit CLI flags
    # win over recipe-set defaults. See src/nla/training/recipes.py.
    from nla.training.recipes import V7_SFT_DEFAULTS, apply_recipe_defaults
    apply_recipe_defaults(
        parser,
        argv if argv is not None else sys.argv[1:],
        recipes={"v7": V7_SFT_DEFAULTS},
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    # Install signal handlers so external kills are no longer silent. Without
    # this, SIGTERM/SIGINT just stop the process with no log line — making it
    # impossible to tell from logs whether the run crashed or was killed.
    import signal
    def _on_signal(signum, _frame):
        signame = signal.Signals(signum).name
        logging.error(
            "[signal] received %s (signum=%d); exiting. Last successful "
            "checkpoint is whatever lives on disk in --output-dir.",
            signame, signum,
        )
        # Re-raise as SystemExit so finally blocks (TB writer close, etc) run.
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    # SIGHUP can happen if a parent shell exits and nohup isn't in play.
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _on_signal)

    from nla.models import ARConfig, AVConfig
    from nla.training.sft import SFTConfig, run_sft

    position_mix: dict[str, float] | None = None
    if args.position_mix_json:
        try:
            position_mix = json.loads(args.position_mix_json)
        except json.JSONDecodeError as e:
            raise SystemExit(f"--position-mix-json: invalid JSON: {e}") from e
        if not isinstance(position_mix, dict):
            raise SystemExit("--position-mix-json must be a JSON object (dictionary).")
        position_mix = {str(k): float(v) for k, v in position_mix.items()}

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
        av_prompt_version=args.av_prompt_version,
        av_include_step_index=not args.av_no_step_index,
        av_include_instruction=not args.av_no_instruction,
        av_num_image_slots=int(args.av_num_image_slots),
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
        ar_prompt_version=args.ar_prompt_version,
        ar_include_step_index=not args.ar_no_step_index,
        ar_include_instruction=not args.ar_no_instruction,
        head_type=args.ar_head_type,
        spatial_n_positions=int(args.ar_spatial_n_positions),
        spatial_head_hidden=int(args.ar_spatial_head_hidden),
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
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
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
        position_mix=position_mix,
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
        exclude_position_types=(
            tuple(args.exclude_position_types) if args.exclude_position_types else None
        ),
        include_position_types=(
            tuple(args.include_position_types) if args.include_position_types else None
        ),
        action_consistency_weight=args.action_consistency_weight,
        action_consistency_every_n_steps=args.action_consistency_every_n_steps,
        action_consistency_max_microbatch=args.action_consistency_max_microbatch,
        action_consistency_image_patch_only=args.action_consistency_image_patch_only,
        action_consistency_blend=args.action_consistency_blend,
        batch_stratified_positions=args.batch_stratified_positions,
        action_consistency_policy_path=args.action_consistency_policy_path,
        action_consistency_embodiment_tag=args.action_consistency_embodiment_tag,
        action_consistency_dataset_roots=(
            json.loads(args.action_consistency_dataset_roots)
            if args.action_consistency_dataset_roots else None
        ),
        action_consistency_manifest_cache=args.action_consistency_manifest_cache,
        action_consistency_suites=(
            tuple(args.action_consistency_suites)
            if args.action_consistency_suites else None
        ),
    )
    summary = run_sft(cfg)
    logging.info("SFT done. %s", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
