"""v7 training recipes: full retrain defaults for SFT and GRPO.

The v7 retrain plan addresses the Stage-0 finding that AR injection at trained
alpha is inert (matched == mismatched == no_steer == 62.5%). The root cause is
that the SFT loss optimizes round-trip h, not policy behavior. v7 reweights the
objective so policy-effect dominates and makes GRPO's contrastive + null arms
mandatory rather than optional.

Usage::

    # In a CLI script:
    from nla.training.recipes import apply_recipe_defaults
    parser = _build_parser()
    apply_recipe_defaults(parser, sys.argv[1:])
    args = parser.parse_args()

Explicit CLI flags always win over recipe defaults: the recipe only changes
the argparse defaults, so an explicit ``--beta 0.01`` overrides v7's 0.05.

External requirements left for the user to provide
--------------------------------------------------
The recipe cannot default these because they're paths to specific artifacts:

SFT v7 requires:
    --action-consistency-policy-path     frozen GR00T checkpoint
    --action-consistency-embodiment-tag  e.g. LIBERO_PANDA
    --action-consistency-dataset-roots   JSON mapping suite -> lerobot root
    --ar-spatial-n-positions             must match GR00T's image_patch count

GRPO v7 requires:
    --sft-dir                            from a v7 SFT run
    --sim-counterfactual-pairs-path      from mine_grpo_counterfactual_pairs.py
    a running NlaSteerGr00tPolicy server on --sim-policy-host:--sim-policy-port

Mapping of every v7 setting to the failure it addresses lives in
``docs/sft_plan/v7_runbook.md``.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# v7 SFT defaults
# ---------------------------------------------------------------------------

V7_SFT_DEFAULTS: dict[str, Any] = {
    # --- Policy-effect loss: promote action_consistency to primary ---------
    # Optional flag becomes mandatory. weight=1.0 means MSE(action) competes
    # with reconstruction MSE on the loss. Drop every-n-steps to 1 so every
    # batch contributes policy-grounded gradient. Disable image-patch-only
    # so last_text and anchor positions also see policy signal.
    "action_consistency_weight": 1.0,
    "action_consistency_every_n_steps": 1,
    "action_consistency_image_patch_only": False,
    # Train at blend=0.5 so the steering hook is the same configuration at
    # train and eval time. Stage 0 showed the policy was OOD at trained
    # alpha (hard replace) and again at half alpha (blend=0.5). Training
    # at blend=0.5 makes that configuration in-distribution.
    "action_consistency_blend": 0.5,

    # --- Down-weight reconstruction; let policy-effect dominate ------------
    # The MSE-on-h objective is what produced the inert codec. Keep it as a
    # regularizer but don't let it set the optimum.
    "ar_weight": 0.1,
    # AV CE on teacher captions stays at 1.0 — it's cheap supervision and
    # provides a sanity-check anchor against templated output.
    "av_weight": 1.0,
    # Contrastive AR NCE term: keep at 0.3 (was 0.0). Hard negatives help
    # break template collapse on image_patch.
    "ar_contrastive_weight": 0.3,
    "ar_nce_hard_negative_source": "topk_cosine",

    # --- Spatial AR head: one vector per image_patch slot ------------------
    # Solves the architectural one-caption-many-patches mismatch.
    # spatial_n_positions must be set by the user to match GR00T's
    # image_patch token count post-pooling (typically 8 with
    # strided_image_multi pooling).
    "ar_head_type": "spatial",
    "image_patch_pooling": "strided_image_multi",
    "image_patch_pooling_strided_k": 8,

    # --- Close train/eval distribution gap ---------------------------------
    # Scheduled sampling: AR sees AV-generated text 70% of the time after
    # warmup (was 30%). Stops AR from over-fitting to gold captions.
    "ar_av_mix_max": 0.7,
    "ar_av_mix_warmup_frac": 0.3,  # Shorter warmup; ramp earlier.
    "ar_av_mix_sample": True,        # Sample from AV (was greedy) — diversity.

    # --- Batch-level position stratification -------------------------------
    # WeightedRandomSampler operates at the epoch level; a single batch can
    # still be 95% last_text by chance. Force per-batch quotas.
    "balance_position_mix": True,
    "batch_stratified_positions": True,

    # --- Prompt + context ---------------------------------------------------
    "av_prompt_version": "context_v5",
    "ar_prompt_version": "context_v5",
    "ar_av_mix_max_new_tokens": 128,

    # --- Training schedule --------------------------------------------------
    # Longer schedule because policy-effect loss has more parameters to
    # shape and learns slower than vanilla MSE. The action-consistency
    # forward is the new bottleneck per-step.
    "total_steps": 4000,
    "warmup_steps": 100,
    "learning_rate": 5e-5,    # Slightly lower than 1e-4 default — more terms.
    "batch_size": 4,
    "grad_accum_steps": 2,    # Effective batch 8 to stabilize gradients.

    # --- Eval ---------------------------------------------------------------
    "eval_every": 100,
    "save_every": 500,
    "eval_closed_loop": True,
    "held_out_fraction": 0.05,
    "split_by": "episode",
}


# ---------------------------------------------------------------------------
# v7 GRPO defaults
# ---------------------------------------------------------------------------

V7_GRPO_DEFAULTS: dict[str, Any] = {
    # --- Sim reward dominates; reconstruction nearly off -------------------
    # The base recon reward is the same loss as SFT — optimizing it adds
    # variance without information. Spend GRPO on the policy signal SFT
    # couldn't reach.
    "sim_reward_weight": 0.8,
    "judge_reward_weight": 0.15,
    # ``ar_co_train_weight`` is the GRPO analog of "ar_weight" — keep small.
    "ar_co_train_weight": 0.05,

    # --- Mandatory contrastive + null arms ---------------------------------
    # Contrastive: r += w * (succ_matched - succ_mismatched). Isolates
    # steering from task difficulty.
    "sim_contrastive_weight": 1.0,
    # Null control: r += w * (succ_matched - succ_null). Rules out
    # "AV won by producing high-magnitude generic noise."
    "sim_null_control_weight": 0.5,
    # Use the language-swap protocol so train and eval contrastive arms
    # measure the same thing.
    "sim_eval_protocol": "language_swap",
    # Intent-conditioned AV prompt: AV learns to write text targeted at a
    # specific task, not just describe the scene.
    "no_intent_conditioned_prompt": False,

    # --- Group size: cut Bernoulli noise -----------------------------------
    # T1-fast used B=2 K=2 → sigma ~0.25 on advantage. B=4 K=8 → sigma
    # ~0.09. The contrast_gap walked downward because half the gradient
    # updates pointed the wrong way.
    "batch_size": 4,
    "rollouts_per_activation": 8,

    # --- KL leash on by default --------------------------------------------
    # T1-fast had beta=0 — no anchor against drift. 30 noisy steps left
    # the SFT basin. 0.05 is small enough to allow learning, big enough
    # to prevent drift.
    "beta": 0.05,
    # Don't disable the KL anchor (which skips compute entirely).
    "disable_kl_anchor": False,

    # --- Lower learning rate, more steps -----------------------------------
    "learning_rate": 1e-6,
    "warmup_steps": 30,
    "total_steps": 300,

    # --- Curriculum (easy → hard CF pairs) ---------------------------------
    # Hard pairs early give all-zero rewards (both arms fail) → all-zero
    # advantages → no learning. Easy pairs first establish gradient
    # direction. Difficulty must be precomputed and stored as a
    # ``difficulty: float`` field on CF pair rows; see
    # ``scripts/training/score_cf_pair_difficulty.py``.
    "curriculum_easy_to_hard": True,

    # --- Sim placement / blend ---------------------------------------------
    # Match the SFT training-time blend so train and eval sim share a
    # configuration.
    "sim_placement": "image_patch",
    "sim_blend": 0.5,
    "sim_w_predicate": 1.5,
    "sim_max_steps": 100,

    # --- Misc stability ----------------------------------------------------
    "dynamic_sampling": True,            # Drop zero-advantage groups.
    "advantage_clip": 5.0,                # Catch outliers.
    "grad_clip": 1.0,

    # --- Eval ---------------------------------------------------------------
    "eval_every": 25,
    "save_every": 50,
    "save_step_snapshots": True,
    "eval_max_examples": 32,
    "held_out_fraction": 0.05,
    "split_by": "episode",
}


def apply_recipe_defaults(
    parser: argparse.ArgumentParser,
    argv: list[str] | None,
    *,
    recipe_arg: str = "--recipe",
    recipes: dict[str, dict[str, Any]],
) -> str | None:
    """Apply a recipe's defaults to ``parser`` before final parse.

    Looks at ``argv`` for ``--recipe <name>``. If found and the name is in
    ``recipes``, calls ``parser.set_defaults(**recipes[name])`` so that the
    final ``parser.parse_args()`` returns recipe values for any flag the
    user did not pass explicitly. Returns the recipe name (or None).

    The function does NOT consume the ``--recipe`` flag — it must still be
    declared on ``parser`` so the final ``parse_args`` accepts it.
    """
    if argv is None:
        return None

    recipe_name: str | None = None
    for i, tok in enumerate(argv):
        if tok == recipe_arg and i + 1 < len(argv):
            recipe_name = argv[i + 1]
            break
        if tok.startswith(recipe_arg + "="):
            recipe_name = tok.split("=", 1)[1]
            break

    if recipe_name is None:
        return None

    if recipe_name not in recipes:
        valid = sorted(recipes)
        raise SystemExit(
            f"Unknown recipe {recipe_name!r}. Valid: {valid}."
        )

    defaults = recipes[recipe_name]
    parser.set_defaults(**defaults)
    logger.info(
        "Applied recipe %r with %d defaults. Explicit CLI flags still win.",
        recipe_name, len(defaults),
    )
    return recipe_name


def v7_sft_required_external() -> list[str]:
    """The v7 SFT recipe still requires these external paths from the user."""
    return [
        "--action-consistency-policy-path",
        "--action-consistency-embodiment-tag",
        "--action-consistency-dataset-roots",
        "--ar-spatial-n-positions",
    ]


def v7_grpo_required_external() -> list[str]:
    """The v7 GRPO recipe still requires these external paths from the user."""
    return [
        "--sft-dir",
        "--sim-counterfactual-pairs-path",
        "--sim-policy-host (default localhost)",
        "--sim-policy-port (default 5555)",
    ]
