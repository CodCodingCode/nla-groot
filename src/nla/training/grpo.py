"""GRPO RL fine-tuning for the Verbalizer (AV) with the Reconstructor (AR) as reward.

Plan-bible recipe (from the NLA paper, adapted for GR00T):

    reward(h, y) = -||h_alpha - AR(y)||^2        (in alpha-scaled space)

where ``y`` is a stochastic verbalization sampled from the *current* AV
policy and ``h_alpha = h / alpha`` is the alpha-scaled ground-truth activation
(matching AR's output convention so the L2 lives in a well-conditioned space
regardless of the chosen alpha).

For each batch of activations ``h_1..h_B`` we draw ``K`` rollouts per
activation, score each rollout under the current policy and a *frozen* SFT
reference, compute group-relative advantages (each group = the K rollouts of
the same activation), and minimize::

    L = -E[A_i^k * log pi(y_i^k | h_i)]  +  beta * KL_token(pi || pi_ref)

The KL term uses Schulman's k3 estimator (unbiased, non-negative)::

    log_ratio = log pi(y) - log pi_ref(y)
    kl_token  = exp(log_ratio) - 1 - log_ratio

We start with AR frozen at its SFT state. Phase 5b can re-enable AR co-training
(joint optimizer with a weighted MSE on the warm-start labels) but the paper
shows the bulk of FVE gains come from the AV side alone, so we hold AR fixed
for the first runs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from nla.models import ActivationReconstructor, ActivationVerbalizer
from nla.training.checkpoint import load_ar_from_sft, load_av_from_sft
from nla.training.dataset import SampledPositionDataset, collate_sampled_positions
from nla.training.fve import StratifiedFve

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------


@dataclass
class GRPOConfig:
    """GRPO trainer configuration.

    Required:
        sft_dir:           Path to an SFT output directory (with ``av/`` and ``ar/``
                           subdirs). Used to initialize both the policy AV and the
                           frozen reference AV, plus the frozen reward model AR.
        activations_root:  Phase-1 extraction root (for SampledPositionDataset).
        output_dir:        Where to write checkpoints / metrics / tb logs.
    """

    sft_dir: str = ""
    activations_root: str = ""
    output_dir: str = ""

    seed: int = 0
    device: str = "cuda"

    # Sampling
    batch_size: int = 4            # B: distinct activations per step
    # V3 default raised from 4 to 8 (see docs/sft_plan/02_hyperparams.md
    # "V3 defaults"): a larger group-relative reward population makes the
    # advantage normalization less noisy at the cost of K× rollout-time
    # memory + compute. CLI users can still pass --rollouts-per-activation
    # 4 to recover the V2 group size.
    rollouts_per_activation: int = 8  # K
    rollout_max_new_tokens: int = 160
    rollout_temperature: float = 1.0
    rollout_top_p: float = 0.95

    # GRPO hyperparameters
    beta: float = 0.02             # KL coefficient (paper says sweep {0.01, 0.02, 0.05})
    advantage_normalize: bool = True
    advantage_clip: float | None = None   # optional advantage clipping
    # When True (default), divide per-group advantages by the group's reward
    # std (after mean-centering). When False, only mean-center within each
    # group—useful when K is small or std estimates are unstable.
    reward_normalize_groupwise: bool = True
    grad_accum_steps: int = 1
    grad_clip: float = 1.0

    # Optimizer
    learning_rate: float = 3e-6    # GRPO is much more sensitive than SFT
    warmup_steps: int = 20
    total_steps: int = 200
    weight_decay: float = 0.0

    # Logging / eval
    eval_every: int = 25
    save_every: int = 100
    log_every: int = 1
    eval_max_examples: int = 64

    # Misc
    gradient_checkpointing: bool = False
    drop_last: bool = False
    held_out_fraction: float = 0.05
    position_mix: dict | None = None  # passthrough to SampledPositionDataset
    # "episode" = hold out whole episodes (default; needed for the
    # memorization-vs-generalization metric).  "row" = legacy random row split.
    split_by: str = "episode"

    # When ``split_by == "episode"`` but the dump only has one episode (or
    # missing ``episode_index``), normally we warn and fall back to row split.
    # Set to ``False`` for paper / generalization runs to fail loudly instead.
    allow_episode_split_row_fallback: bool = True

    # Evaluation: list of temperatures to sample rollouts at during val. The
    # gap between greedy (0.0) and sampled FVE is itself a memorization
    # diagnostic -- memorized AVs have very low entropy.
    eval_temperatures: tuple[float, ...] = (0.0, 0.7, 1.0)

    # Loss switches (mostly for ablations / smoke tests)
    use_kl: bool = True
    use_pg: bool = True

    # AR co-training. When > 0, AR is unfrozen and trained alongside AV on
    # the same rollouts: ar_loss = MSE(AR(rollout), h / alpha).  This closes
    # the open-loop / closed-loop gap by letting AR adapt to AV's evolving
    # output distribution instead of staying pinned at the warm-start labels.
    ar_co_train_weight: float = 0.0

    # Optional multimodal-judge reward term (Workstream B).  When
    # ``judge_reward_weight > 0`` we blend ``r_recon`` (z-scored within the
    # step) with ``r_judge`` from the existing GPT-5.1 grader fed the same
    # cached camera frames the labeler saw.  Default 0 = pure reconstruction
    # (byte-identical to pre-judge runs; the new fields are even hidden from
    # the saved ``config.json``).
    judge_reward_weight: float = 0.0
    judge_concurrency: int = 8
    judge_model: str | None = None
    judge_cache_path: str | None = None
    frames_cache: str | None = None
    # Camera-key tokens used to construct per-row image filenames as
    # ``{frames_cache}/{source_example_id}__{video_key}.jpg``.  Required (and
    # validated as non-empty) whenever ``judge_reward_weight > 0``.  LIBERO
    # runs use ``["image", "wrist_image"]``; any corpus that adheres to the
    # flat ``{source_id}__{key}.jpg`` cache layout works with arbitrary tokens.
    judge_video_keys: list[str] = field(default_factory=list)

    # Optional sim-success reward (Framing B sim-GRPO).  When
    # ``sim_reward_weight > 0`` we encode each AV rollout text with the AR,
    # dispatch a short LIBERO rollout per (activation, text) pair against a
    # long-running NlaSteerGr00tPolicy server, score the rollout with the
    # custom predicate + dense shaping in nla.eval.steerability.predicates,
    # and blend the result into the reward as a third term. Default 0 = pure
    # reconstruction / judge (byte-identical to pre-sim runs; the new fields
    # are hidden from the saved ``config.json``).
    sim_reward_weight: float = 0.0
    sim_counterfactual_pairs_path: str | None = None
    # Optional extra CF pairs files merged into the primary sampler index.
    # Useful when a run wants to plug in mined slices for additional shards
    # (e.g. spatial + object atop a primary goal file) without having to
    # hand-merge the JSONLs offline. Rows are deduped per id-bucket on
    # ``(source_example_id, target_intent, target_task, target_env_name)``
    # so a pair appearing in multiple files isn't double-weighted.
    sim_counterfactual_pairs_paths_extra: list[str] = field(default_factory=list)
    # When True, restore the legacy all-or-nothing batch sim gate: if any
    # row in a batch is missing a valid ``(target_task, target_env_name)``
    # CF pair, sim reward is zeroed for the entire step. Default False
    # enables per-row sim eligibility: sim is computed for rows that do
    # have a pair and skipped (z-mean only over eligible entries) for
    # those that don't, so partial-coverage batches still learn from sim.
    sim_require_full_batch_cf: bool = False
    # When set, ``SampledPositionDataset`` keeps only activations whose
    # ``example_id`` appears in this manifest (built by
    # ``scripts/training/build_grpo_cf_manifest.py`` from the CF pairs
    # JSONLs). Avoids GRPO steps that skip sim because the batch sampled
    # rows with no mined counterfactual pair.
    cf_eligible_ids_path: str | None = None
    sim_policy_host: str = "localhost"
    sim_policy_port: int = 5555
    sim_n_workers: int = 18
    # Rollouts per batched LIBERO subprocess (requires NlaPolicyServer
    # ``get_action_batch``). 4 = one GPU forward per 4 envs per sim step.
    sim_batch_size: int = 4
    sim_max_steps: int = 100
    sim_placement: str = "image_patch"
    sim_blend: float = 1.0
    sim_cache_path: str | None = None
    sim_rollout_python: str | None = None
    sim_rollout_script: str | None = None
    sim_timeout_s: float = 240.0
    # Seed used to derive per-rollout sim seeds. The actual per-rollout seed
    # is ``sim_seed_base + step * 9973 + i`` so each step + rollout-index
    # gets a unique but reproducible env reset.
    sim_seed_base: int = 0
    # When True, AV rollouts are conditioned on the per-row target intent
    # text from the counterfactual pairs file (uses
    # AV_PROMPT_INTENT_CONDITIONED_TEMPLATE). Set to False for pure
    # reconstruction GRPO where the AV describes the activation freely.
    use_intent_conditioned_prompt: bool = True

    # ------------------------------------------------------------------
    # SimpleVLA-RL-inspired knobs (all OFF/default for byte-identical
    # baseline runs; the new fields are hidden from the saved
    # ``config.json`` whenever they sit at their default values).
    # ------------------------------------------------------------------
    #
    # 1) DYNAMIC SAMPLING. SimpleVLA-RL drops groups whose binary rollout
    # rewards have zero variance (the GRPO advantage collapses to 0 anyway
    # so they contribute no learning signal but still cost compute + KL
    # mass). Our rewards are continuous (recon MSE, judge ±1.5, sim shape)
    # so the analog is to drop any group whose per-group reward std is
    # below ``dynamic_sampling_threshold``. The mask zeros out the row's
    # contribution to PG loss, KL loss, and the reported reward stats so
    # collapsed groups don't bias the diagnostics either. Default:
    # ``None`` -> auto-enable when ``sim_reward_weight > 0`` (sim rewards
    # are binary-ish and collapse often), otherwise OFF. Set explicitly
    # to ``True`` / ``False`` from the CLI to override the auto-rule.
    dynamic_sampling: bool | None = None
    dynamic_sampling_threshold: float = 1e-4

    # 2) CLIP-HIGHER. PPO-style importance-ratio clipping with separate
    # low/high bounds (SimpleVLA-RL paper §3.2). Our trainer takes a
    # single gradient step per rollout batch, so the importance ratio
    # ``exp(new_logprobs - new_logprobs.detach())`` is identically 1 at
    # the gradient eval point and the clip is a no-op for one-step
    # updates. The form still matters when ``grad_accum_steps > 1`` (the
    # ratio drifts from 1 within an accumulation window once the very
    # first .backward() has flowed grad into the params -- though only
    # after .step(), so even there the effect is tiny in our setup).
    # We keep the surface for parity with SimpleVLA-RL and so we can
    # later add multi-epoch updates per rollout without reshuffling
    # the loss code. Default OFF preserves byte-identical loss math.
    use_ppo_clip: bool = False
    clip_eps_low: float = 0.2
    clip_eps_high: float = 0.28

    # 3) NO-KL MODE. SimpleVLA-RL ablates the KL anchor entirely on the
    # premise that long-horizon policy entropy collapses anyway and the
    # ref-AV memory + score_tokens forward is wasted. When True, we skip
    # both the ref-policy logprob computation and the KL term in the
    # loss; ``_build_models`` also avoids loading ``ref_av`` so we save
    # the memory. Default False (keeps the paper-bible KL anchor).
    disable_kl_anchor: bool = False

    # 4) ROLLOUT TEMPERATURE OVERRIDE. SimpleVLA-RL anneals rollout
    # temperature on a curriculum; for now we just expose a single high
    # override: when set, ``grpo_step`` uses this value instead of
    # ``rollout_temperature``. We can layer a curriculum on top later.
    rollout_temperature_high: float | None = None


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _serialize_config(cfg: GRPOConfig) -> dict[str, Any]:
    d = asdict(cfg)
    # Hide judge fields entirely when the feature is off so a baseline run's
    # config.json is byte-identical to the pre-judge layout.
    if cfg.judge_reward_weight <= 0.0:
        for k in (
            "judge_reward_weight", "judge_concurrency", "judge_model",
            "judge_cache_path", "frames_cache", "judge_video_keys",
        ):
            d.pop(k, None)
    if cfg.sim_reward_weight <= 0.0:
        for k in (
            "sim_reward_weight", "sim_counterfactual_pairs_path",
            "sim_counterfactual_pairs_paths_extra",
            "sim_require_full_batch_cf", "cf_eligible_ids_path",
            "sim_policy_host", "sim_policy_port", "sim_n_workers", "sim_batch_size",
            "sim_max_steps", "sim_placement", "sim_blend",
            "sim_cache_path", "sim_rollout_python", "sim_rollout_script",
            "sim_timeout_s", "sim_seed_base", "use_intent_conditioned_prompt",
        ):
            d.pop(k, None)
    else:
        # When sim is on, hide the extras/flag fields if they're at default
        # so an old-style config.json stays comparable to existing logs.
        if not cfg.sim_counterfactual_pairs_paths_extra:
            d.pop("sim_counterfactual_pairs_paths_extra", None)
        if not cfg.sim_require_full_batch_cf:
            d.pop("sim_require_full_batch_cf", None)
        if not cfg.cf_eligible_ids_path:
            d.pop("cf_eligible_ids_path", None)
    # SimpleVLA-RL knobs are hidden whenever they sit at their defaults
    # so a baseline run's config.json stays byte-identical to the
    # pre-SimpleVLA layout. Each is dropped independently.
    if cfg.dynamic_sampling is None:
        d.pop("dynamic_sampling", None)
    if cfg.dynamic_sampling_threshold == 1e-4:
        d.pop("dynamic_sampling_threshold", None)
    if not cfg.use_ppo_clip:
        d.pop("use_ppo_clip", None)
    if cfg.clip_eps_low == 0.2:
        d.pop("clip_eps_low", None)
    if cfg.clip_eps_high == 0.28:
        d.pop("clip_eps_high", None)
    if not cfg.disable_kl_anchor:
        d.pop("disable_kl_anchor", None)
    if cfg.rollout_temperature_high is None:
        d.pop("rollout_temperature_high", None)
    if cfg.reward_normalize_groupwise:
        d.pop("reward_normalize_groupwise", None)
    return d


def _setup_outputs(out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "av": out_dir / "av",
        "ar": out_dir / "ar",
        "log": out_dir / "log",
        "metrics": out_dir / "metrics.jsonl",
        "config": out_dir / "config.json",
        "rollouts": out_dir / "rollouts.jsonl",
    }
    paths["av"].mkdir(exist_ok=True)
    paths["ar"].mkdir(exist_ok=True)
    paths["log"].mkdir(exist_ok=True)
    return paths


def _lr_schedule(step: int, cfg: GRPOConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
    prog = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    return 0.5 * cfg.learning_rate * (1.0 + math.cos(math.pi * min(1.0, prog)))


def _write_jsonl_row(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


# ----------------------------------------------------------------------------
# Multimodal judge reward (optional)
# ----------------------------------------------------------------------------


def _judge_cache_key(
    source_id: str,
    rollout_text: str,
    grader_model: str = "",
) -> str:
    """SHA1 cache id for (source_id, text); include grader_model when set.

    Appending the resolved judge model avoids reusing verdicts after changing
    ``--judge-model`` / OPENAI_GRADER_MODEL. When ``grader_model`` is empty,
    the hash matches the legacy two-field layout (tests / backward compat).
    """
    h = hashlib.sha1()
    h.update(source_id.encode("utf-8"))
    h.update(b":")
    h.update(rollout_text.encode("utf-8"))
    if grader_model:
        h.update(b":")
        h.update(grader_model.encode("utf-8"))
    return h.hexdigest()


def _load_judge_cache(path: str | Path | None) -> dict[str, dict]:
    """Read an append-only JSONL cache of judge verdicts into a dict.

    Missing / unreadable files yield an empty dict.  Caller is expected to
    pass the same dict back into ``_compute_judge_rewards`` so it gets
    mutated in place across steps.
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[str, dict] = {}
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            key = obj.get("key")
            if key:
                out[key] = obj
    return out


def _image_paths_for_source(
    source_id: str,
    frames_cache: Path,
    video_keys: list[str],
) -> list[str]:
    """Return cached frame paths for ``source_id`` against ``video_keys``.

    Mirrors ``scripts/eval/llm_judge_av_captions.py:_image_paths_for`` so the
    labeling pipeline, the standalone judge, and GRPO's online judge reward
    all agree on the ``{frames_cache}/{source_id}__{video_key}.jpg`` filename
    convention. Missing files are silently skipped per key; an empty return
    list means the caller should treat the rollout as having no visual
    grounding and fall back to ``r_judge = 0``.
    """
    paths: list[str] = []
    for key in video_keys:
        candidate = frames_cache / f"{source_id}__{key}.jpg"
        if candidate.exists():
            paths.append(str(candidate))
    return paths


def _verdicts_to_scalar(grounding: str | None, appropriateness: str | None) -> float:
    """Map judge verdicts onto r_judge ∈ {-1.5, -0.5, +0.5, +1.5} (0 on error)."""
    if grounding is None or appropriateness is None:
        return 0.0
    b = +1.0 if grounding == "specific" else -1.0
    c = +0.5 if appropriateness == "appropriate" else -0.5
    return b + c


def _blend_rewards(
    r_recon: torch.Tensor,
    r_judge: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    """Z-score r_recon within the step then blend with r_judge.

    When ``weight <= 0`` we return ``r_recon`` untouched (no z-scoring, no
    extra allocations) so baseline runs are byte-identical.
    """
    if weight <= 0.0:
        return r_recon
    mean = r_recon.mean()
    std = r_recon.std().clamp_min(1e-6)
    r_recon_norm = (r_recon - mean) / std
    return (1.0 - weight) * r_recon_norm + weight * r_judge.to(r_recon)


def _zscore(x: torch.Tensor) -> torch.Tensor:
    if x.numel() <= 1:
        return x - x.mean()
    return (x - x.mean()) / x.std().clamp_min(1e-6)


def _blend_multi_rewards(
    r_recon: torch.Tensor,
    *,
    r_judge: torch.Tensor | None = None,
    judge_weight: float = 0.0,
    r_sim: torch.Tensor | None = None,
    sim_weight: float = 0.0,
    sim_active: torch.Tensor | None = None,
) -> torch.Tensor:
    """Blend up to three reward terms (recon + judge + sim).

    Behavior contract -- in priority order:

      * Both weights == 0  →  return ``r_recon`` untouched (byte-identical to
        pre-judge/pre-sim code; this is what unit-tested baseline runs hit).
      * Only judge weight > 0  →  legacy ``_blend_rewards(r_recon, r_judge, judge_weight)``
        path (recon z-scored, judge in its native ±1.5 scale).
      * Sim weight > 0  →  three-way convex blend over z-scored terms:

            (1 - judge_weight - sim_weight) * z(r_recon)
              + judge_weight * z(r_judge or 0)
              + sim_weight   * z(r_sim)

        We z-score the sim term too because its raw scale (predicate ∈ {0,1}
        + ≈[0,1] dense shaping) is incommensurable with recon (negative MSE
        on alpha-scaled activations). Group-relative advantages survive this
        z-scoring, and the resulting term sits in the same numerical regime
        as the other two so a single LR works for all three blends.

    When ``sim_active`` is provided (a 0/1 mask over ``r_sim`` indices), it
    encodes per-row sim eligibility (partial-coverage sim-GRPO batches:
    rows without a counterfactual pair get ``sim_active[i] = 0``). The
    z-score of ``r_sim`` is then taken **only over active entries** so
    inactive rows can't pollute the mean/std, and the per-index effective
    sim weight is ``sim_weight * sim_active[i]`` — inactive rows fall back
    to a pure (recon + judge) blend with the recon coefficient bumped back
    up by the recovered sim slack. ``sim_active is None`` is byte-identical
    to the pre-partial blend.
    """
    if judge_weight <= 0.0 and sim_weight <= 0.0:
        return r_recon

    # Legacy two-way blend path.
    if sim_weight <= 0.0:
        assert r_judge is not None
        return _blend_rewards(r_recon, r_judge, judge_weight)

    if r_sim is None:
        raise ValueError("sim_weight > 0 requires r_sim")
    if judge_weight > 0.0 and r_judge is None:
        raise ValueError("judge_weight > 0 requires r_judge")

    r_sim = r_sim.to(r_recon)
    if r_judge is not None:
        r_judge = r_judge.to(r_recon)

    # Classic all-rows-active path. Keep the existing scalar arithmetic so
    # baseline runs are byte-identical to the pre-partial implementation.
    if sim_active is None:
        base = 1.0 - judge_weight - sim_weight
        if base < 0.0:
            logger.warning(
                "judge_weight (%.3f) + sim_weight (%.3f) > 1; recon weight "
                "clamped to 0",
                judge_weight, sim_weight,
            )
            base = 0.0
        out = base * _zscore(r_recon)
        if judge_weight > 0.0:
            out = out + judge_weight * _zscore(r_judge)
        out = out + sim_weight * _zscore(r_sim)
        return out

    # Partial-coverage path: sim contributes only on active rows; recon
    # picks up the slack on inactive ones so total reward magnitude stays
    # comparable across rows.
    sim_active = sim_active.to(r_recon)
    if sim_active.shape != r_sim.shape:
        raise ValueError(
            f"sim_active shape {tuple(sim_active.shape)} must match "
            f"r_sim shape {tuple(r_sim.shape)}"
        )

    # Per-row effective sim coefficient + per-row base (recon) coefficient.
    w_eff_sim = sim_weight * sim_active
    base_per_row = torch.clamp(1.0 - judge_weight - w_eff_sim, min=0.0)
    if (1.0 - judge_weight - sim_weight) < 0.0:
        logger.warning(
            "judge_weight (%.3f) + sim_weight (%.3f) > 1; recon weight "
            "clamped to 0 on active rows",
            judge_weight, sim_weight,
        )

    # z-score sim only over active entries; inactive entries contribute 0
    # to the sum because w_eff_sim is 0 there.
    if sim_active.sum() > 0:
        active_mask = sim_active.bool()
        active_vals = r_sim[active_mask]
        if active_vals.numel() > 1:
            mu = active_vals.mean()
            sigma = active_vals.std().clamp_min(1e-6)
        else:
            mu = active_vals.mean() if active_vals.numel() == 1 else torch.zeros((), device=r_recon.device, dtype=r_recon.dtype)
            sigma = torch.ones((), device=r_recon.device, dtype=r_recon.dtype)
        z_sim = (r_sim - mu) / sigma
        # Sanitize inactive entries so downstream NaNs from cache hits or
        # error sentinels never sneak in even when w_eff_sim is 0.
        z_sim = torch.where(sim_active.bool(), z_sim, torch.zeros_like(z_sim))
    else:
        z_sim = torch.zeros_like(r_sim)

    out = base_per_row * _zscore(r_recon)
    if judge_weight > 0.0:
        out = out + judge_weight * _zscore(r_judge)
    out = out + w_eff_sim * z_sim
    return out


async def _grade_rollouts_async(
    inputs: list,  # list[GradeInput]
    *,
    model: str,
    concurrency: int,
    max_retries: int = 4,
    base_backoff: float = 1.0,
):
    """Concurrent judge calls; returns the GradeResult per input (order preserved)."""
    from nla.labeling.grader import _grade_one_async
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    sem = asyncio.Semaphore(max(1, concurrency))
    try:
        results = await asyncio.gather(*[
            _grade_one_async(client, inp, model, sem, max_retries, base_backoff)
            for inp in inputs
        ])
    finally:
        await client.close()
    return results


def _compute_judge_rewards(
    rollout_texts: list[str],
    source_example_ids: list[str],
    position_types: list[str],
    *,
    frames_cache: str | Path,
    video_keys: list[str],
    judge_cache: dict[str, dict] | None,
    judge_cache_path: str | Path | None = None,
    judge_model: str | None = None,
    judge_concurrency: int = 8,
    grade_fn=None,
) -> list[float]:
    """Score each rollout with the multimodal judge, returning r_judge values.

    Side-effects:
      * Mutates ``judge_cache`` (key -> entry dict) in place.
      * Appends new entries to ``judge_cache_path`` (JSONL) when set.

    ``grade_fn`` is an injection seam for tests: if given, it's called as
    ``grade_fn(inputs, model=..., concurrency=...)`` and must return a list
    of objects exposing ``.grounding`` and ``.appropriateness`` AxisGrade
    attributes (or ``None``).  Production uses the default OpenAI path.
    """
    from nla.labeling.grader import DEFAULT_GRADER_MODEL, GradeInput

    assert len(rollout_texts) == len(source_example_ids) == len(position_types)
    if judge_cache is None:
        judge_cache = {}
    frames_cache = Path(frames_cache)
    if judge_model is not None:
        os.environ["OPENAI_GRADER_MODEL"] = judge_model
    model = judge_model or DEFAULT_GRADER_MODEL

    rewards: list[float] = [0.0] * len(rollout_texts)
    to_grade_indices: list[int] = []
    to_grade_inputs: list = []  # list[GradeInput]
    to_grade_keys: list[str] = []

    for i, (text, src_id, ptype) in enumerate(
        zip(rollout_texts, source_example_ids, position_types)
    ):
        key = _judge_cache_key(src_id, text, grader_model=model)
        cached = judge_cache.get(key)
        if cached is not None:
            rewards[i] = float(cached.get("r_judge", 0.0))
            continue
        ipaths = _image_paths_for_source(src_id, frames_cache, video_keys)
        if not ipaths:
            # No frames -> neutral signal, do not call the grader.
            rewards[i] = 0.0
            continue
        inp = GradeInput(
            example_id=f"{src_id}__grpo__{i}",
            variant_id="grpo_rollout",
            description=text,
            instruction="",
            position_type=ptype,
            image_paths=ipaths,
            seq_len=None,
            position_index=None,
        )
        to_grade_indices.append(i)
        to_grade_inputs.append(inp)
        to_grade_keys.append(key)

    if to_grade_inputs:
        if grade_fn is not None:
            results = grade_fn(
                to_grade_inputs, model=model, concurrency=judge_concurrency,
            )
        else:
            try:
                results = asyncio.run(_grade_rollouts_async(
                    to_grade_inputs, model=model, concurrency=judge_concurrency,
                ))
            except Exception as e:
                logger.warning("Judge grading raised %r; falling back to neutral", e)
                results = [None] * len(to_grade_inputs)

        cache_path = Path(judge_cache_path) if judge_cache_path else None
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
        fout = cache_path.open("a") if cache_path is not None else None
        try:
            for idx, key, res in zip(to_grade_indices, to_grade_keys, results):
                grounding = None
                appropriateness = None
                if res is not None:
                    g = getattr(res, "grounding", None)
                    a = getattr(res, "appropriateness", None)
                    grounding = getattr(g, "verdict", None) if g is not None else None
                    appropriateness = getattr(a, "verdict", None) if a is not None else None
                r = _verdicts_to_scalar(grounding, appropriateness)
                rewards[idx] = r
                entry = {
                    "key": key,
                    "source_id": source_example_ids[idx],
                    "rollout_text": rollout_texts[idx],
                    "verdict_b": grounding,
                    "verdict_c": appropriateness,
                    "r_judge": r,
                }
                judge_cache[key] = entry
                if fout is not None:
                    fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    fout.flush()
        finally:
            if fout is not None:
                fout.close()

    return rewards


def _build_gen_mask(
    gen_ids: torch.Tensor,
    eos_id: int,
    pad_id: int,
) -> torch.Tensor:
    """Return (B, T_gen) mask: 1 for real generated tokens up to (and including)
    the first EOS, 0 after that or on pad. If a row has no EOS it is treated as
    fully generated.
    """
    B, T = gen_ids.shape
    mask = torch.zeros_like(gen_ids, dtype=torch.long)
    for b in range(B):
        end = T
        for t in range(T):
            tid = int(gen_ids[b, t].item())
            if tid == eos_id:
                end = t + 1  # include the EOS itself
                break
            if tid == pad_id and pad_id != eos_id:
                end = t
                break
        mask[b, :end] = 1
    return mask


# ----------------------------------------------------------------------------
# Core GRPO step
# ----------------------------------------------------------------------------


def grpo_step(
    policy_av: ActivationVerbalizer,
    ref_av: ActivationVerbalizer | None,
    ar: ActivationReconstructor,
    activations: torch.Tensor,
    position_types: list[str],
    *,
    K: int,
    beta: float,
    rollout_max_new_tokens: int,
    rollout_temperature: float,
    rollout_top_p: float,
    advantage_normalize: bool = True,
    advantage_clip: float | None = None,
    use_kl: bool = True,
    use_pg: bool = True,
    ar_train_weight: float = 0.0,
    source_example_ids: list[str] | None = None,
    frames_cache: str | None = None,
    judge_video_keys: list[str] | None = None,
    judge_reward_weight: float = 0.0,
    judge_cache: dict[str, dict] | None = None,
    judge_cache_path: str | None = None,
    judge_model: str | None = None,
    judge_concurrency: int = 8,
    judge_grade_fn=None,
    target_intent_texts: list[str] | None = None,
    target_tasks: list[str] | None = None,
    target_env_names: list[str] | None = None,
    sim_reward_weight: float = 0.0,
    sim_worker=None,                # SimRewardWorker | None
    sim_seeds: list[int] | None = None,
    sim_max_steps: int = 100,
    sim_placement: str = "image_patch",
    sim_blend: float = 1.0,
    sim_cf_ok: list[bool] | None = None,
    # ----- SimpleVLA-RL-inspired knobs (defaults preserve byte-identical loss math)
    dynamic_sampling: bool = False,
    dynamic_sampling_threshold: float = 1e-4,
    use_ppo_clip: bool = False,
    clip_eps_low: float = 0.2,
    clip_eps_high: float = 0.28,
    disable_kl_anchor: bool = False,
    reward_normalize_groupwise: bool = True,
) -> dict:
    """One GRPO update worth of forward computation (no optimizer step here).

    Returns a dict with the loss (call ``.backward()`` outside) plus a
    handful of detached diagnostics for logging.

    When ``ar_train_weight > 0``, AR is run with grad and the MSE term
    ``ar_train_weight * MSE(AR(rollouts), h / alpha)`` is added to the loss.
    Caller is responsible for putting AR params in the optimizer.
    """
    device = activations.device
    B = activations.shape[0]
    assert len(position_types) == B
    pad_id = policy_av._pad_id
    eos_id = (
        policy_av.tokenizer.eos_token_id
        if policy_av.tokenizer.eos_token_id is not None
        else pad_id
    )

    # ----- 1. Expand each activation K times for grouped rollouts ------------
    acts_rep = activations.repeat_interleave(K, dim=0).to(device)        # (B*K, H)
    ptypes_rep: list[str] = [p for p in position_types for _ in range(K)]
    intents_rep: list[str | None] | None = None
    if target_intent_texts is not None:
        if len(target_intent_texts) != B:
            raise ValueError(
                f"len(target_intent_texts)={len(target_intent_texts)} != B={B}"
            )
        intents_rep = [t for t in target_intent_texts for _ in range(K)]

    # ----- 2. Rollout under current policy (no grad) -------------------------
    policy_av.eval()
    with torch.no_grad():
        rollout = policy_av.generate(
            acts_rep,
            ptypes_rep,
            max_new_tokens=rollout_max_new_tokens,
            temperature=rollout_temperature,
            top_p=rollout_top_p,
            do_sample=True,
            return_logprobs=False,
            target_intent_texts=intents_rep,
        )
    gen_ids = rollout["token_ids"]                                       # (B*K, T_gen)
    gen_mask = _build_gen_mask(gen_ids, eos_id=eos_id, pad_id=pad_id)
    rollout_texts = rollout["text"]                                      # len B*K

    # ----- 3. Reward = -|| (h/alpha) - AR(y) ||^2  (mean over hidden dim) ----
    # When ar_train_weight > 0 we run AR with grad and reuse the same forward
    # for both the (detached) reward and the differentiable MSE co-training
    # term. Otherwise AR is in eval mode and the forward is no-grad.
    if ar_train_weight > 0.0:
        ar.train()
        pred_scaled = ar(rollout_texts, device=device)                   # (B*K, H_act)  WITH grad
        target_scaled = (acts_rep / ar.cfg.alpha).to(pred_scaled.dtype)
        rewards = -((pred_scaled.detach() - target_scaled.detach()) ** 2).mean(dim=-1).float()
        ar_mse = ((pred_scaled - target_scaled) ** 2).mean()             # scalar, with grad
    else:
        ar.eval()
        with torch.no_grad():
            pred_scaled = ar(rollout_texts, device=device)
            target_scaled = (acts_rep / ar.cfg.alpha).to(pred_scaled.dtype)
            rewards = -((pred_scaled - target_scaled) ** 2).mean(dim=-1).float()
        ar_mse = torch.zeros((), device=device)

    # ----- 3b. Optional multimodal-judge blend -------------------------------
    # Default weight=0.0 short-circuits before any judge work; baseline runs
    # are byte-identical to pre-judge code.
    judge_rewards_list: list[float] | None = None
    r_judge_tensor: torch.Tensor | None = None
    if judge_reward_weight > 0.0 and frames_cache is not None:
        if source_example_ids is None:
            raise ValueError(
                "judge_reward_weight > 0 requires source_example_ids "
                "(plumbed from the dataset)."
            )
        if len(source_example_ids) != B:
            raise ValueError(
                f"len(source_example_ids)={len(source_example_ids)} != B={B}"
            )
        if not judge_video_keys:
            raise ValueError(
                "judge_reward_weight > 0 requires a non-empty judge_video_keys "
                "list (e.g. ['image', 'wrist_image'] for LIBERO)."
            )
        src_rep = [s for s in source_example_ids for _ in range(K)]
        judge_rewards_list = _compute_judge_rewards(
            rollout_texts=rollout_texts,
            source_example_ids=src_rep,
            position_types=ptypes_rep,
            frames_cache=frames_cache,
            video_keys=judge_video_keys,
            judge_cache=judge_cache,
            judge_cache_path=judge_cache_path,
            judge_model=judge_model,
            judge_concurrency=judge_concurrency,
            grade_fn=judge_grade_fn,
        )
        r_judge_tensor = torch.tensor(
            judge_rewards_list, dtype=rewards.dtype, device=rewards.device,
        )

    # ----- 3c. Optional sim-success reward (Framing B sim-GRPO) --------------
    # Default weight=0.0 short-circuits before any sim work; baseline runs are
    # byte-identical to pre-sim code. When enabled, we encode each rollout
    # text through the *frozen* AR -> backbone-space steer vector ``hhat``,
    # then dispatch a LIBERO rollout per row via ``sim_worker``.
    #
    # Per-row eligibility (``sim_cf_ok``) lets us partial-blend batches where
    # only some activations have a counterfactual pair: ineligible rows still
    # appear in ``target_tasks``/``target_env_names``/``source_example_ids``
    # (placeholder strings are fine, they're only used by sim jobs) but are
    # skipped at job assembly + masked out of the sim blend.
    sim_results = None  # list of SimRewardResult, in original row-major order
    r_sim_tensor: torch.Tensor | None = None
    sim_active_tensor: torch.Tensor | None = None
    if sim_reward_weight > 0.0:
        if sim_worker is None:
            raise ValueError("sim_reward_weight > 0 requires sim_worker")
        if target_tasks is None or len(target_tasks) != B:
            raise ValueError(
                "sim_reward_weight > 0 requires target_tasks of length B"
            )
        if target_env_names is None or len(target_env_names) != B:
            raise ValueError(
                "sim_reward_weight > 0 requires target_env_names of length B"
            )
        if not source_example_ids or len(source_example_ids) != B:
            raise ValueError(
                "sim_reward_weight > 0 requires source_example_ids of length B"
            )
        if sim_seeds is None or len(sim_seeds) != B * K:
            raise ValueError(
                f"sim_seeds must have length B*K={B * K}, got "
                f"{None if sim_seeds is None else len(sim_seeds)}"
            )

        # Per-row sim eligibility (length B). Default: every row eligible
        # (preserves legacy behavior when callers don't pass ``sim_cf_ok``).
        if sim_cf_ok is None:
            sim_cf_ok_b = [True] * B
        else:
            if len(sim_cf_ok) != B:
                raise ValueError(
                    f"len(sim_cf_ok)={len(sim_cf_ok)} != B={B}"
                )
            sim_cf_ok_b = list(sim_cf_ok)

        # Replicate per-activation metadata K times to align with rollout_texts.
        target_tasks_rep = [t for t in target_tasks for _ in range(K)]
        envs_rep = [e for e in target_env_names for _ in range(K)]
        src_rep = [s for s in source_example_ids for _ in range(K)]
        sim_active_rep = [ok for ok in sim_cf_ok_b for _ in range(K)]
        active_idx = [i for i, ok in enumerate(sim_active_rep) if ok]

        # Encode every rollout text into a backbone-space steer vector with
        # the *frozen* AR (so the gradient never flows through the sim
        # subprocess, which obviously cannot be backproped). We encode for
        # ALL rows (cheap forward), but only build SimRewardJobs for the
        # eligible ones.
        from nla.training.sim_reward import (
            assemble_jobs,
            encode_texts_with_ar,
        )
        with torch.no_grad():
            steer_vecs = encode_texts_with_ar(ar, rollout_texts, device=device)

        if active_idx:
            sub_steer = steer_vecs[active_idx]
            sub_texts = [rollout_texts[i] for i in active_idx]
            sub_tasks = [target_tasks_rep[i] for i in active_idx]
            sub_envs = [envs_rep[i] for i in active_idx]
            sub_src = [src_rep[i] for i in active_idx]
            sub_seeds = [sim_seeds[i] for i in active_idx]
            jobs = assemble_jobs(
                rollout_texts=sub_texts,
                steer_vecs=sub_steer,
                target_tasks=sub_tasks,
                target_env_names=sub_envs,
                source_ids=sub_src,
                seeds=sub_seeds,
                sim_max_steps=sim_max_steps,
                placement=sim_placement,
                blend=sim_blend,
            )
            sub_results = sim_worker.compute(jobs)
        else:
            sub_results = []

        # Scatter sub-results back into row-major order; ineligible rows
        # carry ``None`` in ``sim_results`` (diagnostics treat them as
        # skipped) and contribute 0 to the sim term via the mask below.
        sim_results = [None] * (B * K)
        r_sim_vec = [0.0] * (B * K)
        for slot, res in zip(active_idx, sub_results):
            sim_results[slot] = res
            r_sim_vec[slot] = float(res.r_sim)
        r_sim_tensor = torch.tensor(
            r_sim_vec, dtype=rewards.dtype, device=rewards.device,
        )
        sim_active_tensor = torch.tensor(
            [1.0 if ok else 0.0 for ok in sim_active_rep],
            dtype=rewards.dtype, device=rewards.device,
        )

    # ----- 3d. Blend recon + (optional) judge + (optional) sim ---------------
    # Effective judge weight is 0 when the judge block was silently skipped
    # (e.g. frames_cache=None) so we don't accidentally claim a judge term
    # that was never computed.
    eff_judge_weight = judge_reward_weight if r_judge_tensor is not None else 0.0
    rewards = _blend_multi_rewards(
        rewards,
        r_judge=r_judge_tensor,
        judge_weight=eff_judge_weight,
        r_sim=r_sim_tensor,
        sim_weight=sim_reward_weight,
        sim_active=sim_active_tensor,
    )

    # ----- 4. Group-relative advantage ---------------------------------------
    rewards_grp = rewards.view(B, K)
    adv_grp = rewards_grp - rewards_grp.mean(dim=1, keepdim=True)
    group_std = rewards_grp.std(dim=1)                                   # (B,) unmasked
    if advantage_normalize and reward_normalize_groupwise and K > 1:
        std = rewards_grp.std(dim=1, keepdim=True).clamp_min(1e-8)
        adv_grp = adv_grp / std
    advantages = adv_grp.view(B * K)
    if advantage_clip is not None:
        advantages = advantages.clamp(-advantage_clip, advantage_clip)

    # ----- 4b. Dynamic-sampling row mask -------------------------------------
    # SimpleVLA-RL: drop groups with ~zero reward variance (their advantages
    # collapse to 0 anyway; keeping them just dilutes the KL/PG signal and
    # corrupts the per-step reward stats). We do this at the row level
    # (B*K,) as a multiplicative loss mask + a stats mask, instead of
    # poking ``gen_mask`` directly so token_counts stay honest for the
    # per-row mean.
    row_keep = torch.ones(B * K, dtype=rewards.dtype, device=rewards.device)
    drop_frac = 0.0
    if dynamic_sampling:
        keep_grp = (group_std >= dynamic_sampling_threshold).to(rewards.dtype)
        # Expand (B,) -> (B*K,)
        row_keep = keep_grp.unsqueeze(1).expand(B, K).reshape(B * K)
        drop_frac = float((1.0 - row_keep).mean().item())

    # ----- 5. Score under current policy (with grad) and frozen ref (no grad)
    policy_av.train()
    new_logprobs = policy_av.score_tokens(
        acts_rep, ptypes_rep, gen_ids, gen_mask,
        target_intent_texts=intents_rep,
    )

    # When the KL anchor is disabled, skip the ref forward entirely (saves
    # ~one full policy-AV forward pass per step + the ref's GPU memory in
    # ``run_grpo``). ``use_kl`` is the legacy ablation switch that gates
    # only the loss term while still running the ref forward + reporting
    # ``kl_token_mean`` as a diagnostic; ``disable_kl_anchor`` is the
    # stricter SimpleVLA-RL-style "don't even compute the ref logprobs"
    # knob and necessarily zeroes the KL diagnostic too.
    skip_ref = disable_kl_anchor or ref_av is None
    if skip_ref:
        ref_logprobs = None
    else:
        with torch.no_grad():
            ref_av.eval()
            ref_logprobs = ref_av.score_tokens(
                acts_rep, ptypes_rep, gen_ids, gen_mask,
                target_intent_texts=intents_rep,
            )

    mask = gen_mask.to(new_logprobs.dtype)
    token_counts = mask.sum(dim=1).clamp_min(1)                          # (B*K,)

    # ----- 6. Policy-gradient loss + KL --------------------------------------
    if use_pg:
        if use_ppo_clip:
            # PPO clip-higher (SimpleVLA-RL §3.2): split the symmetric PPO
            # clip into asymmetric eps_low / eps_high. Note: our trainer
            # takes one gradient step per rollout batch, so
            # ``new_logprobs.detach()`` IS the "old logprobs" of that step
            # -> ``ratio = exp(0) = 1`` identically at the gradient eval
            # point. The clip therefore is a no-op for one-step updates;
            # we still write the loss in this form so when we later add
            # multi-epoch updates per rollout (the regime where SimpleVLA-RL
            # actually uses clip-higher) the math is already wired up.
            log_ratio_new = new_logprobs - new_logprobs.detach()
            ratio = torch.exp(log_ratio_new)
            adv = advantages.detach().unsqueeze(-1)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps_low, 1.0 + clip_eps_high) * adv
            pg_per_token = -torch.min(surr1, surr2) * mask
        else:
            # Vanilla GRPO PG: -E[A * log pi(y)].
            pg_per_token = -advantages.detach().unsqueeze(-1) * new_logprobs * mask
        pg_per_row = pg_per_token.sum(dim=1) / token_counts              # (B*K,)
        pg_per_row = pg_per_row * row_keep
        denom = row_keep.sum().clamp_min(1.0) if dynamic_sampling else float(B * K)
        pg_loss = pg_per_row.sum() / denom
    else:
        pg_loss = torch.zeros((), device=device)

    if ref_logprobs is not None:
        log_ratio = new_logprobs - ref_logprobs                          # log pi/pi_ref
        log_ratio = log_ratio * mask
        # k3 estimator: nonneg, unbiased for KL(pi || pi_ref).
        kl_per_token = (torch.exp(log_ratio) - 1.0 - log_ratio) * mask
        if use_kl:
            kl_per_row = kl_per_token.sum(dim=1) / token_counts          # (B*K,)
            kl_per_row = kl_per_row * row_keep
            denom = row_keep.sum().clamp_min(1.0) if dynamic_sampling else float(B * K)
            kl_loss = kl_per_row.sum() / denom
        else:
            kl_loss = torch.zeros((), device=device)
    else:
        kl_per_token = torch.zeros_like(new_logprobs)
        kl_loss = torch.zeros((), device=device)

    total_loss = pg_loss + beta * kl_loss + ar_train_weight * ar_mse

    # ----- 7. Diagnostics ----------------------------------------------------
    with torch.no_grad():
        n_tok = mask.sum().clamp_min(1)
        # When dynamic sampling drops groups, the row-keep mask also gates
        # the reward stats so the reported reward_mean/std/best/worst
        # reflect only the rows that actually contributed gradient. Falls
        # back to the original full-batch stats when sampling is off.
        if dynamic_sampling and row_keep.sum().item() > 0:
            keep_bool = row_keep.bool()
            kept = rewards[keep_bool]
            reward_mean = float(kept.mean().item())
            reward_std = (
                float(kept.std(unbiased=False).item()) if kept.numel() > 1 else 0.0
            )
            reward_best = float(kept.max().item())
            reward_worst = float(kept.min().item())
            advantage_abs_mean = float((advantages.abs() * row_keep).sum().item()
                                       / max(1.0, float(row_keep.sum().item())))
        else:
            reward_mean = float(rewards.mean().item())
            reward_std = (
                float(rewards.std(unbiased=False).item()) if rewards.numel() > 1 else 0.0
            )
            reward_best = float(rewards.max().item())
            reward_worst = float(rewards.min().item())
            advantage_abs_mean = float(advantages.abs().mean().item())

        diagnostics = {
            "loss": float(total_loss.item()),
            "pg_loss": float(pg_loss.item()) if use_pg else 0.0,
            "kl_loss": float(kl_loss.item()) if use_kl else 0.0,
            "ar_mse": float(ar_mse.item()) if ar_train_weight > 0.0 else 0.0,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "reward_best": reward_best,
            "reward_worst": reward_worst,
            "advantage_abs_mean": advantage_abs_mean,
            "kl_token_mean": float((kl_per_token.sum() / n_tok).item()),
            "logp_new_mean": float((new_logprobs * mask).sum().item() / float(n_tok.item())),
            "logp_ref_mean": (
                float((ref_logprobs * mask).sum().item() / float(n_tok.item()))
                if ref_logprobs is not None else 0.0
            ),
            "gen_len_mean": float(token_counts.float().mean().item()),
        }
        if dynamic_sampling:
            diagnostics["dynamic_sampling_drop_frac"] = drop_frac
        if judge_rewards_list is not None:
            jt = torch.tensor(judge_rewards_list, dtype=torch.float32)
            diagnostics["judge_reward_mean"] = float(jt.mean().item())
            diagnostics["judge_reward_pos_frac"] = float((jt > 0).float().mean().item())
        if sim_results is not None:
            # Active rows are those where ``sim_results[i]`` is a real
            # SimRewardResult; ineligible rows are stored as ``None``
            # placeholders so the row-index alignment with ``r_sim_vec``
            # / ``sim_active_tensor`` is preserved. Diagnostics aggregate
            # only over the active entries; we additionally report
            # ``sim_active_frac`` so dashboards can see how much of the
            # batch contributed to the sim term this step.
            active_results = [r for r in sim_results if r is not None]
            n_total = len(sim_results)
            n_active = len(active_results)
            diagnostics["sim_active_frac"] = float(n_active) / max(1, n_total)
            if active_results:
                sr = torch.tensor([r.r_sim for r in active_results], dtype=torch.float32)
                pred = torch.tensor([r.predicate for r in active_results], dtype=torch.float32)
                succ = torch.tensor(
                    [float(r.success_any) for r in active_results], dtype=torch.float32,
                )
                n_cached = sum(1 for r in active_results if r.cached)
                n_err = sum(1 for r in active_results if r.error is not None)
                n_early = sum(1 for r in active_results if r.early_stopped)
                elapsed = torch.tensor([r.elapsed_s for r in active_results], dtype=torch.float32)
                diagnostics["sim_reward_mean"] = float(sr.mean().item())
                diagnostics["sim_reward_best"] = float(sr.max().item())
                diagnostics["sim_predicate_pos_frac"] = float((pred > 0).float().mean().item())
                diagnostics["sim_success_any_frac"] = float(succ.mean().item())
                diagnostics["sim_early_stop_frac"] = float(n_early) / n_active
                diagnostics["sim_cache_hit_frac"] = float(n_cached) / n_active
                diagnostics["sim_error_frac"] = float(n_err) / n_active
                diagnostics["sim_elapsed_s_mean"] = float(elapsed.mean().item())

    return {
        "loss": total_loss,
        "pg_loss": pg_loss,
        "kl_loss": kl_loss,
        "ar_mse": ar_mse,
        "rewards": rewards.detach(),
        "advantages": advantages.detach(),
        "rollout_texts": rollout_texts,
        "gen_ids": gen_ids.detach(),
        "gen_mask": gen_mask.detach(),
        "sim_results": sim_results,
        "diagnostics": diagnostics,
    }


# ----------------------------------------------------------------------------
# Evaluation: FVE on greedy AV rollouts
# ----------------------------------------------------------------------------


def _metrics_with_closed_greedy_aliases(
    metrics: dict[str, float],
    *,
    greedy_temperature: float = 0.0,
) -> dict[str, float]:
    """Mirror SFT ``closed_greedy/*`` keys so ``build_v3_scorecard`` can read GRPO val rows.

    SFT logs ``closed_{tag}/{metric}`` from :func:`nla.training.sft._evaluate` with
    ``tag="greedy"`` when ``temperature <= 0``.  GRPO's :func:`_evaluate_fve` emits
    ``{metric}/temp=0.0`` instead.  We duplicate greedy-temperature scalars under
    ``closed_greedy/{metric}`` (including stratified keys like
    ``fve/position=image_patch`` → ``closed_greedy/fve/position=image_patch``).
    """
    suffix = f"/temp={greedy_temperature}"
    out = dict(metrics)
    for k, v in metrics.items():
        if not k.endswith(suffix):
            continue
        stem = k[: -len(suffix)]
        out[f"closed_greedy/{stem}"] = v
    return out


@torch.no_grad()
def _evaluate_fve(
    policy_av: ActivationVerbalizer,
    ar: ActivationReconstructor,
    loader,
    device: str | torch.device,
    *,
    max_examples: int,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 160,
    temperatures: tuple[float, ...] | None = None,
) -> dict:
    """Sample one rollout per held-out activation and measure FVE against AR.

    This is the closed-loop NLA reconstruction metric: AV explains, AR
    reconstructs, FVE measures variance explained.  When ``temperatures`` is
    provided we run one full pass per temperature and prefix metric keys with
    ``temp=<t>``; the legacy single-temperature path is preserved when
    ``temperatures is None``.

    Per-position-type metrics are emitted under keys like
    ``fve/position=image_patch`` -- this is the slice that argues NLAs are
    uniquely valuable on the backbone-image positions, where SAE features
    have no native readout.
    """
    policy_av.eval()
    ar.eval()
    temps = temperatures if temperatures is not None else (temperature,)
    out: dict[str, float] = {}
    for ti, t in enumerate(temps):
        fve_acc = StratifiedFve(group_name="position")
        seen = 0
        do_sample = t > 0.0
        for batch in loader:
            if seen >= max_examples:
                break
            acts = batch["activations"].to(device)
            ptypes = batch["position_type"]
            rollout = policy_av.generate(
                acts, ptypes,
                max_new_tokens=max_new_tokens,
                temperature=t if do_sample else 1.0,
                top_p=top_p,
                do_sample=do_sample,
                return_logprobs=False,
            )
            pred_scaled = ar(rollout["text"], device=device)
            pred_unscaled = pred_scaled.float() * ar.cfg.alpha
            fve_acc.update(acts.float(), pred_unscaled, ptypes)
            seen += acts.shape[0]
        metrics = fve_acc.compute()
        if temperatures is None:
            # Legacy single-temperature call: no prefix, top-level keys.
            return metrics
        for k, v in metrics.items():
            out[f"{k}/temp={t}"] = v
    return out


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------


def _build_dataloaders(cfg: GRPOConfig):
    allowed_ids = None
    if cfg.cf_eligible_ids_path:
        from nla.training.counterfactual_data import load_grpo_cf_manifest

        allowed_ids = load_grpo_cf_manifest(cfg.cf_eligible_ids_path)
        logger.info(
            "Restricting GRPO pool to %d CF-eligible activations "
            "(manifest=%s)",
            len(allowed_ids), cfg.cf_eligible_ids_path,
        )
    train_ds = SampledPositionDataset(
        cfg.activations_root,
        seed=cfg.seed,
        position_mix=cfg.position_mix,
        held_out_fraction=cfg.held_out_fraction,
        held_out=False,
        split_by=cfg.split_by,
        allow_episode_split_row_fallback=cfg.allow_episode_split_row_fallback,
        allowed_example_ids=allowed_ids,
    )
    val_ds = SampledPositionDataset(
        cfg.activations_root,
        seed=cfg.seed,
        position_mix=cfg.position_mix,
        held_out_fraction=cfg.held_out_fraction,
        held_out=True,
        split_by=cfg.split_by,
        allow_episode_split_row_fallback=cfg.allow_episode_split_row_fallback,
        allowed_example_ids=allowed_ids,
    )
    logger.info("Train pool: %d  Val pool: %d", len(train_ds), len(val_ds))
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_sampled_positions,
        drop_last=cfg.drop_last,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_sampled_positions,
        drop_last=False,
    )
    return train_loader, val_loader, train_ds, val_ds


def _build_models(cfg: GRPOConfig):
    sft_dir = Path(cfg.sft_dir)
    logger.info("Loading policy AV from %s", sft_dir / "av")
    policy_av = load_av_from_sft(sft_dir / "av", device=cfg.device, freeze=False)

    # KL_ENABLED: when ``disable_kl_anchor`` is True we skip loading the
    # reference AV entirely (saves the second policy-sized weight set in
    # GPU memory + the per-step ref forward). All KL-touching code paths
    # below check ``ref_av is None`` so this is safe.
    if cfg.disable_kl_anchor:
        logger.info(
            "disable_kl_anchor=True; skipping reference AV load (saves a "
            "policy-AV-sized memory + the per-step ref forward)."
        )
        ref_av = None
    else:
        logger.info("Loading reference AV (frozen) from %s", sft_dir / "av")
        ref_av = load_av_from_sft(sft_dir / "av", device=cfg.device, freeze=True)

    ar_frozen = cfg.ar_co_train_weight <= 0.0
    logger.info(
        "Loading AR (%s) from %s",
        "frozen" if ar_frozen else "trainable, co-train weight=%g" % cfg.ar_co_train_weight,
        sft_dir / "ar",
    )
    ar = load_ar_from_sft(sft_dir / "ar", device=cfg.device, freeze=ar_frozen)

    if cfg.gradient_checkpointing:
        for fn in ("gradient_checkpointing_enable", "enable_input_require_grads"):
            if hasattr(policy_av.base, fn):
                try:
                    getattr(policy_av.base, fn)()
                except Exception as e:
                    logger.warning("Could not %s on policy AV: %s", fn, e)
    return policy_av, ref_av, ar


def _validate_judge_config(cfg: GRPOConfig) -> None:
    """Fail fast when the judge term is requested but its prerequisites are missing."""
    if cfg.judge_reward_weight <= 0.0:
        return
    if not cfg.frames_cache:
        raise ValueError(
            "judge_reward_weight > 0 requires --frames-cache (directory of "
            "cached camera frames named {source_id}__{video_key}.jpg)."
        )
    if not cfg.judge_video_keys:
        raise ValueError(
            "judge_reward_weight > 0 requires --judge-video-keys (camera-key "
            "tokens, e.g. 'image wrist_image' for LIBERO). Frame filenames "
            "are resolved as {frames_cache}/{source_id}__{video_key}.jpg."
        )
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "judge_reward_weight > 0 requires OPENAI_API_KEY in the environment."
        )


def _validate_sim_config(cfg: GRPOConfig) -> None:
    """Fail fast when sim-reward GRPO is requested but its prerequisites are missing."""
    if cfg.sim_reward_weight <= 0.0:
        return
    if not cfg.sim_counterfactual_pairs_path:
        raise ValueError(
            "sim_reward_weight > 0 requires --sim-counterfactual-pairs-path "
            "(JSONL produced by scripts/training/mine_grpo_counterfactual_pairs.py)."
        )
    if not Path(cfg.sim_counterfactual_pairs_path).exists():
        raise ValueError(
            f"sim_counterfactual_pairs_path does not exist: "
            f"{cfg.sim_counterfactual_pairs_path}"
        )
    for extra in cfg.sim_counterfactual_pairs_paths_extra:
        if not Path(extra).exists():
            raise ValueError(
                f"sim_counterfactual_pairs_paths_extra entry does not exist: {extra}"
            )
    if cfg.sim_n_workers < 1:
        raise ValueError("sim_n_workers must be >= 1")
    if not (0.0 <= cfg.sim_blend <= 1.0):
        raise ValueError(f"sim_blend must be in [0, 1]; got {cfg.sim_blend}")
    if cfg.sim_max_steps < 1:
        raise ValueError(f"sim_max_steps must be >= 1; got {cfg.sim_max_steps}")


def run_grpo(cfg: GRPOConfig) -> dict:
    _validate_judge_config(cfg)
    _validate_sim_config(cfg)
    out_dir = Path(cfg.output_dir)
    paths = _setup_outputs(out_dir)
    paths["config"].write_text(json.dumps(_serialize_config(cfg), indent=2))
    torch.manual_seed(cfg.seed)

    train_loader, val_loader, train_ds, _ = _build_dataloaders(cfg)
    if len(train_ds) == 0:
        raise RuntimeError("No training activations; check activations_root.")

    policy_av, ref_av, ar = _build_models(cfg)
    trainable = [p for p in policy_av.parameters() if p.requires_grad]
    n_av_trainable = sum(p.numel() for p in trainable)
    if cfg.ar_co_train_weight > 0.0:
        ar_params = [p for p in ar.parameters() if p.requires_grad]
        trainable = trainable + ar_params
        n_ar_trainable = sum(p.numel() for p in ar_params)
        logger.info(
            "Trainable params: policy AV %.2fM + AR %.2fM = %.2fM",
            n_av_trainable / 1e6, n_ar_trainable / 1e6,
            (n_av_trainable + n_ar_trainable) / 1e6,
        )
    else:
        logger.info("Trainable params (policy AV): %d (~%.2fM)", n_av_trainable, n_av_trainable / 1e6)

    optim = torch.optim.AdamW(
        trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )

    tb = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(str(paths["log"]))
    except Exception:
        logger.info("tensorboard unavailable; skipping TB logging.")

    step = 0
    accum_count = 0
    optim.zero_grad(set_to_none=True)
    start = time.time()
    final_metrics: dict = {}

    judge_cache = (
        _load_judge_cache(cfg.judge_cache_path)
        if cfg.judge_reward_weight > 0.0 else None
    )
    if judge_cache is not None:
        logger.info(
            "Judge reward enabled (weight=%.3f, concurrency=%d, model=%s); "
            "loaded %d cached verdicts from %s",
            cfg.judge_reward_weight, cfg.judge_concurrency,
            cfg.judge_model or "(default)", len(judge_cache), cfg.judge_cache_path,
        )

    # Resolve SimpleVLA-RL knobs that depend on other config values.
    # ``dynamic_sampling=None`` (the cfg default) means "auto": ON when
    # the sim reward is in the blend (binary-ish rewards collapse often)
    # and OFF otherwise. Explicit True/False overrides the auto-rule.
    dynamic_sampling_eff = (
        bool(cfg.dynamic_sampling)
        if cfg.dynamic_sampling is not None
        else (cfg.sim_reward_weight > 0.0)
    )
    rollout_temp_eff = (
        cfg.rollout_temperature_high
        if cfg.rollout_temperature_high is not None
        else cfg.rollout_temperature
    )
    if cfg.rollout_temperature_high is not None:
        logger.info(
            "Using rollout_temperature_high=%.3f override (was %.3f)",
            cfg.rollout_temperature_high, cfg.rollout_temperature,
        )
    if dynamic_sampling_eff:
        logger.info(
            "Dynamic sampling ON (threshold=%.2e); groups with reward std "
            "below threshold are dropped from PG/KL/reward-stats.",
            cfg.dynamic_sampling_threshold,
        )
    if cfg.use_ppo_clip:
        logger.info(
            "PPO clip-higher ON (eps_low=%.3f, eps_high=%.3f). Note: a "
            "no-op for one-step updates per rollout (ratio == 1).",
            cfg.clip_eps_low, cfg.clip_eps_high,
        )
    if cfg.disable_kl_anchor:
        logger.info("KL anchor disabled (no ref-AV scoring, no KL term in loss).")

    sim_worker = None
    cf_sampler = None
    if cfg.sim_reward_weight > 0.0:
        from nla.training.counterfactual_data import CounterfactualPairSampler
        from nla.training.sim_reward import SimRewardWorker
        cf_sampler = CounterfactualPairSampler(
            cfg.sim_counterfactual_pairs_path,
            seed=cfg.seed,
            additional_paths=cfg.sim_counterfactual_pairs_paths_extra or None,
        )
        sim_worker = SimRewardWorker(
            policy_host=cfg.sim_policy_host,
            policy_port=cfg.sim_policy_port,
            n_workers=cfg.sim_n_workers,
            sim_batch_size=cfg.sim_batch_size,
            sim_max_steps=cfg.sim_max_steps,
            placement=cfg.sim_placement,
            blend=cfg.sim_blend,
            rollout_python=cfg.sim_rollout_python,
            rollout_script=cfg.sim_rollout_script,
            cache_path=cfg.sim_cache_path,
            timeout_s=cfg.sim_timeout_s,
        )
        logger.info(
            "Sim reward enabled (weight=%.3f, n_workers=%d, batch_size=%d, "
            "max_steps=%d, placement=%s, blend=%.2f); counterfactual pairs "
            "loaded from %s (%d source_ids covered)",
            cfg.sim_reward_weight, cfg.sim_n_workers, cfg.sim_batch_size,
            cfg.sim_max_steps, cfg.sim_placement, cfg.sim_blend,
            cfg.sim_counterfactual_pairs_path, len(cf_sampler),
        )

    while step < cfg.total_steps:
        for batch in train_loader:
            if step >= cfg.total_steps:
                break

            for g in optim.param_groups:
                g["lr"] = _lr_schedule(step, cfg)

            acts = batch["activations"].to(cfg.device, non_blocking=True)
            ptypes = batch["position_type"]
            # source_ids are needed whenever the judge OR sim reward is on,
            # plus any other downstream lookups (counterfactual mining).
            needs_source_ids = (
                cfg.judge_reward_weight > 0.0 or cfg.sim_reward_weight > 0.0
            )
            source_ids = batch.get("example_id") if needs_source_ids else None

            # Look up a fresh counterfactual (intent, env_name, target_task)
            # per activation. ``sample_for`` returns a sentinel pair (empty
            # target_task / env_name) for ids missing from the JSONL. We
            # build a per-row ``sim_cf_ok`` mask so partial-coverage
            # batches still learn from sim on the rows that do have a
            # pair; ``sim_require_full_batch_cf`` (CLI flag) restores the
            # legacy all-or-nothing batch gate.
            target_intent_texts = None
            target_tasks = None
            target_env_names = None
            sim_seeds = None
            sim_cf_ok: list[bool] | None = None
            effective_sim_weight = cfg.sim_reward_weight
            if cfg.sim_reward_weight > 0.0 and cf_sampler is not None:
                if not source_ids:
                    raise RuntimeError(
                        "sim_reward_weight > 0 needs batch['example_id']; "
                        "got an empty/None list -- check dataset wiring."
                    )
                pairs = cf_sampler.sample_for(source_ids)
                sim_cf_ok = [
                    bool(p.target_task and p.target_env_name) for p in pairs
                ]
                n_ok = sum(sim_cf_ok)
                n_missing = len(sim_cf_ok) - n_ok
                if n_missing:
                    missing_ids = [
                        source_ids[i] for i, ok in enumerate(sim_cf_ok) if not ok
                    ][:3]
                    if cfg.sim_require_full_batch_cf:
                        logger.warning(
                            "[step %d] %d/%d batch rows have no counterfactual "
                            "pair in %s; sim_require_full_batch_cf=True so "
                            "the sim term is skipped this step. First few "
                            "missing: %s",
                            step, n_missing, len(sim_cf_ok),
                            cfg.sim_counterfactual_pairs_path, missing_ids,
                        )
                        effective_sim_weight = 0.0
                    else:
                        logger.info(
                            "[step %d] %d/%d batch rows missing a CF pair; "
                            "computing sim on the remaining %d rows "
                            "(partial blend). First few missing: %s",
                            step, n_missing, len(sim_cf_ok), n_ok, missing_ids,
                        )

                # Always plumb full-length B lists into ``grpo_step``; rows
                # without a CF pair get placeholders that the per-row mask
                # ignores. Intent is ``None`` per-row so AV falls back to
                # the descriptive prompt for those rows.
                target_intent_texts = [
                    p.target_intent if ok else None
                    for p, ok in zip(pairs, sim_cf_ok)
                ]
                target_tasks = [
                    p.target_task if ok else "" for p, ok in zip(pairs, sim_cf_ok)
                ]
                target_env_names = [
                    p.target_env_name if ok else ""
                    for p, ok in zip(pairs, sim_cf_ok)
                ]
                K = cfg.rollouts_per_activation
                sim_seeds = [
                    cfg.sim_seed_base + step * 9973 + i
                    for i in range(len(pairs) * K)
                ]

                # If the entire batch lacks CF pairs (e.g. dataset
                # completely misaligned with the JSONL), there's no work
                # for the sim worker. Drop the sim term to avoid an empty
                # batch.
                if n_ok == 0:
                    effective_sim_weight = 0.0

            # The intent-conditioned AV prompt is independent of sim rollouts:
            # it's how the policy AV learns to *write* intent-targeted text in
            # the first place. We pass intents whenever the cfg switch is on
            # AND we managed to look them up; otherwise fall back to the
            # legacy descriptive prompt.
            av_intent_texts = (
                target_intent_texts if cfg.use_intent_conditioned_prompt else None
            )

            out = grpo_step(
                policy_av, ref_av, ar,
                acts, ptypes,
                K=cfg.rollouts_per_activation,
                beta=cfg.beta,
                rollout_max_new_tokens=cfg.rollout_max_new_tokens,
                rollout_temperature=rollout_temp_eff,
                rollout_top_p=cfg.rollout_top_p,
                advantage_normalize=cfg.advantage_normalize,
                advantage_clip=cfg.advantage_clip,
                use_kl=cfg.use_kl,
                use_pg=cfg.use_pg,
                ar_train_weight=cfg.ar_co_train_weight,
                source_example_ids=source_ids,
                frames_cache=cfg.frames_cache,
                judge_video_keys=cfg.judge_video_keys,
                judge_reward_weight=cfg.judge_reward_weight,
                judge_cache=judge_cache,
                judge_cache_path=cfg.judge_cache_path,
                judge_model=cfg.judge_model,
                judge_concurrency=cfg.judge_concurrency,
                target_intent_texts=av_intent_texts,
                target_tasks=target_tasks,
                target_env_names=target_env_names,
                sim_reward_weight=effective_sim_weight,
                sim_worker=sim_worker,
                sim_seeds=sim_seeds,
                sim_max_steps=cfg.sim_max_steps,
                sim_placement=cfg.sim_placement,
                sim_blend=cfg.sim_blend,
                sim_cf_ok=sim_cf_ok,
                dynamic_sampling=dynamic_sampling_eff,
                dynamic_sampling_threshold=cfg.dynamic_sampling_threshold,
                use_ppo_clip=cfg.use_ppo_clip,
                clip_eps_low=cfg.clip_eps_low,
                clip_eps_high=cfg.clip_eps_high,
                disable_kl_anchor=cfg.disable_kl_anchor,
                reward_normalize_groupwise=cfg.reward_normalize_groupwise,
            )
            loss = out["loss"]
            (loss / max(1, cfg.grad_accum_steps)).backward()
            accum_count += 1

            if accum_count >= cfg.grad_accum_steps:
                torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
                optim.step()
                optim.zero_grad(set_to_none=True)
                accum_count = 0

            if step % cfg.log_every == 0:
                row = {
                    "step": step,
                    "phase": "train",
                    **out["diagnostics"],
                    "lr": optim.param_groups[0]["lr"],
                    "elapsed_s": time.time() - start,
                }
                _write_jsonl_row(paths["metrics"], row)
                if tb is not None:
                    for k, v in out["diagnostics"].items():
                        tb.add_scalar(f"train/{k}", v, step)
                    tb.add_scalar("train/lr", row["lr"], step)

            if step > 0 and step % cfg.eval_every == 0:
                eval_metrics = _evaluate_fve(
                    policy_av, ar, val_loader, cfg.device,
                    max_examples=cfg.eval_max_examples,
                    temperature=cfg.rollout_temperature,
                    top_p=cfg.rollout_top_p,
                    max_new_tokens=cfg.rollout_max_new_tokens,
                    temperatures=cfg.eval_temperatures,
                )
                if eval_metrics:
                    eval_metrics = _metrics_with_closed_greedy_aliases(
                        eval_metrics, greedy_temperature=0.0,
                    )
                    row = {
                        "step": step, "phase": "val",
                        **eval_metrics, "elapsed_s": time.time() - start,
                    }
                    _write_jsonl_row(paths["metrics"], row)
                    logger.info(
                        "[step %d] val %s",
                        step,
                        "  ".join(f"{k}={v:.4f}" for k, v in eval_metrics.items()),
                    )
                    if tb is not None:
                        for k, v in eval_metrics.items():
                            tb.add_scalar(f"val/{k}", v, step)
                    final_metrics = eval_metrics

            if step > 0 and step % cfg.save_every == 0:
                policy_av.save(str(paths["av"]))
                if cfg.ar_co_train_weight > 0.0:
                    ar.save(str(paths["ar"]))

            step += 1

    # Final eval + save
    eval_metrics = _evaluate_fve(
        policy_av, ar, val_loader, cfg.device,
        max_examples=cfg.eval_max_examples,
        temperature=cfg.rollout_temperature,
        top_p=cfg.rollout_top_p,
        max_new_tokens=cfg.rollout_max_new_tokens,
        temperatures=cfg.eval_temperatures,
    )
    if eval_metrics:
        eval_metrics = _metrics_with_closed_greedy_aliases(
            eval_metrics, greedy_temperature=0.0,
        )
        row = {"step": step, "phase": "final", **eval_metrics, "elapsed_s": time.time() - start}
        _write_jsonl_row(paths["metrics"], row)
        logger.info("[final] val %s", "  ".join(f"{k}={v:.4f}" for k, v in eval_metrics.items()))
        final_metrics = eval_metrics

    policy_av.save(str(paths["av"]))
    if cfg.ar_co_train_weight > 0.0:
        ar.save(str(paths["ar"]))
    if tb is not None:
        tb.close()
    return {"steps": step, "metrics": final_metrics, "out_dir": str(out_dir)}
