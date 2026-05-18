#!/usr/bin/env python
"""GRPO RL fine-tuning for the AV (Verbalizer).

Loads a warm-start SFT checkpoint, freezes it as the reference policy and as
the reward model (AR), and trains the policy AV with::

    reward(h, y) = -|| (h / alpha) - AR(y) ||^2

KL anchor to the frozen reference at coefficient ``--beta``.

Example (LIBERO, recon-only)::

    PYTHONPATH=src python scripts/training/run_grpo.py \\
        --sft-dir          data/sft/libero_goal_pilot_v3 \\
        --activations-root data/activations/libero_goal_pilot \\
        --output-dir       data/grpo/libero_goal_pilot_b002 \\
        --beta 0.02 --total-steps 200 --rollouts-per-activation 8

Example (LIBERO, recon + multimodal-judge blend)::

    OPENAI_API_KEY=sk-... PYTHONPATH=src python scripts/training/run_grpo.py \\
        --sft-dir          data/sft/libero_goal_pilot_v3 \\
        --activations-root data/activations/libero_goal_pilot \\
        --output-dir       data/grpo/libero_goal_pilot_b002_judge \\
        --beta 0.02 --total-steps 200 --rollouts-per-activation 8 \\
        --judge-reward-weight 0.5 \\
        --frames-cache       data/labels/libero_goal_pilot/frames_cache \\
        --judge-video-keys   image wrist_image

Example (LIBERO, sim-success steerability GRPO)::

    # 1) Launch a long-running NlaSteerGr00tPolicy server in another shell:
    #    python scripts/eval/run_gr00t_server_nla_steer.py --ar-dir data/sft/libero_goal_pilot_v3/ar
    # 2) Mine counterfactual (scene, target_intent) pairs:
    #    python scripts/training/mine_grpo_counterfactual_pairs.py \\
    #        --labels      data/labels/libero_4suite_combined/labels.jsonl \\
    #        --output      data/grpo/cf_pairs.jsonl
    # 3) Train:
    PYTHONPATH=src python scripts/training/run_grpo.py \\
        --sft-dir          data/sft/libero_4suite_v3 \\
        --activations-root data/activations/libero_4suite_combined \\
        --output-dir       data/grpo/libero_4suite_v3_sim \\
        --beta 0.02 --total-steps 200 --rollouts-per-activation 8 \\
        --sim-reward-weight 0.5 \\
        --sim-counterfactual-pairs-path data/grpo/cf_pairs.jsonl \\
        --sim-policy-host localhost --sim-policy-port 5555 \\
        --sim-n-workers 4 --sim-max-steps 100 \\
        --sim-placement image_patch --sim-blend 1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sft-dir", required=True,
                   help="Output dir from run_sft.py (must contain av/ and ar/).")
    p.add_argument("--activations-root", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--batch-size", type=int, default=4,
                   help="Distinct activations per step (B).")
    p.add_argument("--rollouts-per-activation", type=int, default=8,
                   help="K: rollouts per activation, the GRPO group size "
                        "(matches GRPOConfig V3 default; pass 4 for legacy).")
    p.add_argument("--rollout-max-new-tokens", type=int, default=160)
    p.add_argument("--rollout-temperature", type=float, default=1.0)
    p.add_argument("--rollout-top-p", type=float, default=0.95)

    p.add_argument("--beta", type=float, default=0.02,
                   help="KL coefficient (paper sweep: {0.01, 0.02, 0.05}).")
    p.add_argument("--no-advantage-normalize", action="store_true")
    p.add_argument(
        "--no-reward-normalize-groupwise",
        action="store_true",
        help="After mean-centering rewards within each GRPO group, skip dividing "
             "advantages by the group's reward std (keeps centering only). "
             "Default ON: divide by per-group std when --no-advantage-normalize "
             "is not set.",
    )
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
    p.add_argument(
        "--position-mix-json",
        default=None,
        metavar="JSON",
        help="Optional JSON object of position-type weights for SampledPositionDataset "
             '(e.g. \'{"last_text":0.4,"image_patch":0.4,"anchor":0.2}\'). '
             "Omit to use nla.layer_spec.POSITION_MIX.",
    )
    p.add_argument("--eval-temperatures", default="0.0,0.7,1.0",
                   help="Comma-separated rollout temperatures for evaluation.  The "
                        "gap between greedy (0.0) and sampled FVE is itself a "
                        "memorization diagnostic.")

    # Optional multimodal-judge reward term (off by default; weight=0 is
    # byte-identical to the pure-reconstruction recipe).
    p.add_argument("--judge-reward-weight", type=float, default=0.0,
                   help="Blend coefficient for the GPT-5.1 multimodal judge reward "
                        "(0 = pure reconstruction, 1 = pure judge). When > 0, "
                        "--frames-cache is required and OPENAI_API_KEY must be set.")
    p.add_argument("--judge-concurrency", type=int, default=8,
                   help="Max concurrent judge API calls per GRPO step.")
    p.add_argument("--judge-model", default=None,
                   help="Override OPENAI_GRADER_MODEL (default: gpt-5.1).")
    p.add_argument("--judge-cache-path", default=None,
                   help="JSONL cache of judge verdicts keyed by "
                        "sha1(source_id:text:grader_model). "
                        "Read on startup, appended to as new (source_id, rollout) "
                        "pairs are scored.")
    p.add_argument("--frames-cache", default=None,
                   help="Directory of cached camera frames "
                        "({source_id}__{video_key}.jpg), same convention as "
                        "scripts/eval/llm_judge_av_captions.py. Populate via "
                        "scripts/eval/extract_label_frames.py.")
    p.add_argument("--judge-video-keys", nargs="+", default=None,
                   help="Camera-key tokens used to construct per-row image "
                        "filenames at {frames_cache}/{source_id}__{video_key}.jpg. "
                        "Required when --judge-reward-weight > 0. For LIBERO "
                        "pass 'image wrist_image'; the tokens must match what "
                        "your labeling pipeline / extract_label_frames.py wrote.")

    # Sim-success reward (Framing B). Off by default; setting weight=0
    # keeps a baseline GRPO run byte-identical to pre-sim code.
    p.add_argument("--sim-reward-weight", type=float, default=0.0,
                   help="Blend coefficient for the LIBERO sim-success reward "
                        "(0 = no sim term, 1 = pure sim). When > 0, requires "
                        "--sim-counterfactual-pairs-path and a running "
                        "NlaSteerGr00tPolicy server reachable at "
                        "--sim-policy-host:--sim-policy-port. The per-rollout "
                        "score combines a binary predicate (e.g. on/under/in) "
                        "with dense shaping from "
                        "nla.eval.steerability.predicates.")
    p.add_argument("--sim-counterfactual-pairs-path", default=None,
                   help="JSONL of {source_example_id, example_id, "
                        "target_intent, target_task, target_env_name, ...} "
                        "rows produced by "
                        "scripts/training/mine_grpo_counterfactual_pairs.py. "
                        "The sampler indexes each row under BOTH "
                        "``source_example_id`` and ``example_id`` so it "
                        "resolves whichever flavor the activation dataset "
                        "yields per batch.")
    p.add_argument("--sim-counterfactual-pairs-path-extra", default=[],
                   action="append",
                   help="Optional extra CF pairs JSONL(s) merged into the "
                        "primary sampler index. Repeat the flag to add "
                        "multiple files. Rows are deduped per id-bucket on "
                        "(source_example_id, target_intent, target_task, "
                        "target_env_name) so a row appearing in multiple "
                        "files isn't double-weighted.")
    p.add_argument("--sim-require-full-batch-cf",
                   dest="sim_require_full_batch_cf",
                   action="store_true", default=False,
                   help="Restore the legacy all-or-nothing batch gate: if "
                        "any row in a batch is missing a valid CF pair, "
                        "skip the sim term for the whole step. Default OFF "
                        "enables per-row sim eligibility (sim is computed "
                        "for rows that have a pair and skipped for those "
                        "that don't, so partial-coverage batches still "
                        "learn from sim).")
    p.add_argument("--sim-policy-host", default="localhost")
    p.add_argument("--sim-policy-port", type=int, default=5555,
                   help="ZMQ port the NlaSteerGr00tPolicy server is listening "
                        "on. The trainer fans out short rollouts to this one "
                        "server; the server should have been launched with the "
                        "same AR your SFT dir holds.")
    p.add_argument("--sim-n-workers", type=int, default=4,
                   help="Number of concurrent in-flight LIBERO rollouts per "
                        "GRPO step. Each worker is one subprocess holding one "
                        "ZMQ client connection.")
    p.add_argument("--sim-max-steps", type=int, default=100,
                   help="Max simulator steps per rollout (capped further by "
                        "early-stop-on-predicate). Short rollouts speed up "
                        "GRPO at the cost of weaker sparse signal; 100 ~= "
                        "30s of robot motion on LIBERO Goal.")
    p.add_argument("--sim-placement", default="image_patch",
                   choices=["last_text", "image_patch", "anchor", "image_patch_all", "fixed"],
                   help="Per-step steer placement sent to the policy server "
                        "via options['steer_spec'].placement.")
    p.add_argument("--sim-blend", type=float, default=1.0,
                   help="Per-step steer blend factor sent to the policy "
                        "server via options['steer_spec'].blend; lambda=1 is "
                        "full overwrite, lambda=0.5 mixes the original "
                        "activation 50/50 with the AR-decoded one.")
    p.add_argument("--sim-cache-path", default=None,
                   help="JSONL cache of sim rewards keyed by sha1(env|task|"
                        "source_id|text|seed|max_steps).")
    p.add_argument("--sim-rollout-python", default=None,
                   help="Python interpreter to invoke the rollout subprocess "
                        "with. Defaults to $NLA_ROLLOUT_PYTHON or the trainer's "
                        "current interpreter. Production runs typically point "
                        "this at the LIBERO venv.")
    p.add_argument("--sim-rollout-script", default=None,
                   help="Path to rollout.py (default: in-tree "
                        "src/nla/eval/steerability/rollout.py).")
    p.add_argument("--sim-timeout-s", type=float, default=240.0,
                   help="Per-rollout subprocess timeout (kill + score 0 on "
                        "timeout). 240s comfortably covers a 100-step "
                        "LIBERO Goal rollout including env reset.")
    p.add_argument("--sim-seed-base", type=int, default=0,
                   help="Per-rollout seed = sim_seed_base + step*9973 + i.")
    p.add_argument("--no-intent-conditioned-prompt", action="store_true",
                   help="By default sim-GRPO uses an intent-conditioned AV "
                        "prompt (see AV_PROMPT_INTENT_CONDITIONED_TEMPLATE) "
                        "so the policy AV learns to write text targeted at a "
                        "task. Pass this to fall back to the descriptive prompt "
                        "for ablations.")

    # ---- SimpleVLA-RL-inspired knobs (off / auto by default) -----------
    p.add_argument(
        "--dynamic-sampling", dest="dynamic_sampling",
        action="store_const", const=True, default=None,
        help="Drop GRPO groups whose reward std is below "
             "--dynamic-sampling-threshold (the advantage collapses to 0 "
             "anyway; SimpleVLA-RL §3.1). Default: auto -> ON when "
             "--sim-reward-weight > 0 (binary-ish sim rewards collapse "
             "often), OFF otherwise.")
    p.add_argument(
        "--no-dynamic-sampling", dest="dynamic_sampling",
        action="store_const", const=False,
        help="Force dynamic sampling OFF, overriding the auto-rule.")
    p.add_argument(
        "--dynamic-sampling-threshold", type=float, default=1e-4,
        help="Per-group reward std below which the group is masked from "
             "the PG, KL, and reward-stat aggregates. Default 1e-4.")

    p.add_argument(
        "--use-ppo-clip", dest="use_ppo_clip",
        action="store_true", default=False,
        help="Apply PPO-style importance-ratio clipping with asymmetric "
             "low/high bounds (SimpleVLA-RL clip-higher). Note: our "
             "trainer takes one gradient step per rollout, so the ratio "
             "is identically 1 at the gradient eval point and the clip "
             "is a no-op for one-step updates -- it becomes meaningful "
             "if/when we add multi-epoch updates per rollout.")
    p.add_argument(
        "--no-use-ppo-clip", dest="use_ppo_clip", action="store_false",
        help="Disable PPO clip (default).")
    p.add_argument(
        "--clip-eps-low", type=float, default=0.2,
        help="PPO clip lower bound when --use-ppo-clip is set (1 - eps_low).")
    p.add_argument(
        "--clip-eps-high", type=float, default=0.28,
        help="PPO clip upper bound when --use-ppo-clip is set (1 + eps_high).")

    p.add_argument(
        "--disable-kl-anchor", action="store_true",
        help="Skip the KL anchor entirely: don't load the frozen "
             "reference AV (saves a policy-AV-sized memory) and drop "
             "the KL term from the loss (SimpleVLA-RL ablation).")

    p.add_argument(
        "--rollout-temperature-high", type=float, default=None,
        help="Override --rollout-temperature with this value when set. "
             "Hook for a future SimpleVLA-RL-style temperature curriculum; "
             "for now it is a single-value override.")

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

    if args.judge_reward_weight > 0.0:
        if not args.frames_cache:
            raise SystemExit(
                "--judge-reward-weight > 0 requires --frames-cache "
                "(directory of cached camera frames named "
                "{source_id}__{video_key}.jpg)."
            )
        if not args.judge_video_keys:
            raise SystemExit(
                "--judge-reward-weight > 0 requires --judge-video-keys "
                "(camera-key tokens, e.g. 'image wrist_image' for LIBERO)."
            )
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit(
                "--judge-reward-weight > 0 requires OPENAI_API_KEY in the environment."
            )

    if args.sim_reward_weight > 0.0 and not args.sim_counterfactual_pairs_path:
        raise SystemExit(
            "--sim-reward-weight > 0 requires --sim-counterfactual-pairs-path "
            "(produced by scripts/training/mine_grpo_counterfactual_pairs.py)."
        )

    position_mix = None
    if args.position_mix_json:
        try:
            position_mix = json.loads(args.position_mix_json)
        except json.JSONDecodeError as e:
            raise SystemExit(f"--position-mix-json: invalid JSON: {e}") from e
        if not isinstance(position_mix, dict):
            raise SystemExit("--position-mix-json must be a JSON object (dictionary).")

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
        reward_normalize_groupwise=not args.no_reward_normalize_groupwise,
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
        position_mix=position_mix,
        split_by=args.split_by,
        allow_episode_split_row_fallback=not args.no_episode_split_fallback,
        eval_temperatures=eval_temps,
        judge_reward_weight=args.judge_reward_weight,
        judge_concurrency=args.judge_concurrency,
        judge_model=args.judge_model,
        judge_cache_path=args.judge_cache_path,
        frames_cache=args.frames_cache,
        judge_video_keys=list(args.judge_video_keys) if args.judge_video_keys else [],
        sim_reward_weight=args.sim_reward_weight,
        sim_counterfactual_pairs_path=args.sim_counterfactual_pairs_path,
        sim_counterfactual_pairs_paths_extra=list(
            args.sim_counterfactual_pairs_path_extra or []
        ),
        sim_require_full_batch_cf=args.sim_require_full_batch_cf,
        sim_policy_host=args.sim_policy_host,
        sim_policy_port=args.sim_policy_port,
        sim_n_workers=args.sim_n_workers,
        sim_max_steps=args.sim_max_steps,
        sim_placement=args.sim_placement,
        sim_blend=args.sim_blend,
        sim_cache_path=args.sim_cache_path,
        sim_rollout_python=args.sim_rollout_python,
        sim_rollout_script=args.sim_rollout_script,
        sim_timeout_s=args.sim_timeout_s,
        sim_seed_base=args.sim_seed_base,
        use_intent_conditioned_prompt=not args.no_intent_conditioned_prompt,
        dynamic_sampling=args.dynamic_sampling,
        dynamic_sampling_threshold=args.dynamic_sampling_threshold,
        use_ppo_clip=args.use_ppo_clip,
        clip_eps_low=args.clip_eps_low,
        clip_eps_high=args.clip_eps_high,
        disable_kl_anchor=args.disable_kl_anchor,
        rollout_temperature_high=args.rollout_temperature_high,
    )
    summary = run_grpo(cfg)
    logging.info("GRPO done. %s", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
