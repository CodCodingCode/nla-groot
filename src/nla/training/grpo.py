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
    reward_normalize_groupwise: bool = True  # always group-relative; this gates std-norm
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


def _judge_cache_key(source_id: str, rollout_text: str) -> str:
    h = hashlib.sha1()
    h.update(source_id.encode("utf-8"))
    h.update(b":")
    h.update(rollout_text.encode("utf-8"))
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
        key = _judge_cache_key(src_id, text)
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
    ref_av: ActivationVerbalizer,
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
        rewards = _blend_rewards(rewards, r_judge_tensor, judge_reward_weight)

    # ----- 4. Group-relative advantage ---------------------------------------
    rewards_grp = rewards.view(B, K)
    adv_grp = rewards_grp - rewards_grp.mean(dim=1, keepdim=True)
    if advantage_normalize and K > 1:
        std = rewards_grp.std(dim=1, keepdim=True).clamp_min(1e-8)
        adv_grp = adv_grp / std
    advantages = adv_grp.view(B * K)
    if advantage_clip is not None:
        advantages = advantages.clamp(-advantage_clip, advantage_clip)

    # ----- 5. Score under current policy (with grad) and frozen ref (no grad)
    policy_av.train()
    new_logprobs = policy_av.score_tokens(acts_rep, ptypes_rep, gen_ids, gen_mask)

    with torch.no_grad():
        ref_av.eval()
        ref_logprobs = ref_av.score_tokens(acts_rep, ptypes_rep, gen_ids, gen_mask)

    mask = gen_mask.to(new_logprobs.dtype)
    token_counts = mask.sum(dim=1).clamp_min(1)                          # (B*K,)

    # ----- 6. Policy-gradient loss + KL --------------------------------------
    # PG: maximize E[A * log pi(y)]; loss is negative.
    pg_per_token = -advantages.detach().unsqueeze(-1) * new_logprobs * mask
    pg_loss = (pg_per_token.sum(dim=1) / token_counts).mean() if use_pg else torch.zeros((), device=device)

    log_ratio = new_logprobs - ref_logprobs                              # log pi/pi_ref
    log_ratio = log_ratio * mask
    # k3 estimator: nonneg, unbiased for KL(pi || pi_ref).
    kl_per_token = (torch.exp(log_ratio) - 1.0 - log_ratio) * mask
    kl_loss = (kl_per_token.sum(dim=1) / token_counts).mean() if use_kl else torch.zeros((), device=device)

    total_loss = pg_loss + beta * kl_loss + ar_train_weight * ar_mse

    # ----- 7. Diagnostics ----------------------------------------------------
    with torch.no_grad():
        n_tok = mask.sum().clamp_min(1)
        diagnostics = {
            "loss": float(total_loss.item()),
            "pg_loss": float(pg_loss.item()) if use_pg else 0.0,
            "kl_loss": float(kl_loss.item()) if use_kl else 0.0,
            "ar_mse": float(ar_mse.item()) if ar_train_weight > 0.0 else 0.0,
            "reward_mean": float(rewards.mean().item()),
            "reward_std": float(rewards.std(unbiased=False).item()) if rewards.numel() > 1 else 0.0,
            "reward_best": float(rewards.max().item()),
            "reward_worst": float(rewards.min().item()),
            "advantage_abs_mean": float(advantages.abs().mean().item()),
            "kl_token_mean": float((kl_per_token.sum() / n_tok).item()),
            "logp_new_mean": float((new_logprobs * mask).sum().item() / float(n_tok.item())),
            "logp_ref_mean": float((ref_logprobs * mask).sum().item() / float(n_tok.item())),
            "gen_len_mean": float(token_counts.float().mean().item()),
        }
        if judge_rewards_list is not None:
            jt = torch.tensor(judge_rewards_list, dtype=torch.float32)
            diagnostics["judge_reward_mean"] = float(jt.mean().item())
            diagnostics["judge_reward_pos_frac"] = float((jt > 0).float().mean().item())

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
        "diagnostics": diagnostics,
    }


# ----------------------------------------------------------------------------
# Evaluation: FVE on greedy AV rollouts
# ----------------------------------------------------------------------------


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
    train_ds = SampledPositionDataset(
        cfg.activations_root,
        seed=cfg.seed,
        position_mix=cfg.position_mix,
        held_out_fraction=cfg.held_out_fraction,
        held_out=False,
        split_by=cfg.split_by,
        allow_episode_split_row_fallback=cfg.allow_episode_split_row_fallback,
    )
    val_ds = SampledPositionDataset(
        cfg.activations_root,
        seed=cfg.seed,
        position_mix=cfg.position_mix,
        held_out_fraction=cfg.held_out_fraction,
        held_out=True,
        split_by=cfg.split_by,
        allow_episode_split_row_fallback=cfg.allow_episode_split_row_fallback,
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


def run_grpo(cfg: GRPOConfig) -> dict:
    _validate_judge_config(cfg)
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

    while step < cfg.total_steps:
        for batch in train_loader:
            if step >= cfg.total_steps:
                break

            for g in optim.param_groups:
                g["lr"] = _lr_schedule(step, cfg)

            acts = batch["activations"].to(cfg.device, non_blocking=True)
            ptypes = batch["position_type"]
            source_ids = batch.get("example_id") if cfg.judge_reward_weight > 0.0 else None

            out = grpo_step(
                policy_av, ref_av, ar,
                acts, ptypes,
                K=cfg.rollouts_per_activation,
                beta=cfg.beta,
                rollout_max_new_tokens=cfg.rollout_max_new_tokens,
                rollout_temperature=cfg.rollout_temperature,
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
