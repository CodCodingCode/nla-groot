"""Joint warm-start SFT for AV and AR.

Trains both modules on (activation, position_type, description) triples from a
``LabeledPositionDataset``:

- AV: cross-entropy on (activation -> description) via single-slot injection.
- AR: MSE in alpha-scaled space on (description -> activation / alpha).

Joint loss::

    loss = av_weight * CE(AV)  +  ar_weight * MSE(AR)

Single optimizer over all trainable parameters in both modules (LoRA adapters,
the AV activation projector, the AR affine head).

Outputs::

    <output_dir>/
      av/                 # peft save_pretrained + act_proj.pt + av_config.json
      ar/                 # peft save_pretrained + head.pt + ar_config.json
      config.json         # SFTConfig snapshot
      log/                # tensorboard scalars
      metrics.jsonl       # per-step train/val rows
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader

from nla.layer_spec import POSITION_MIX
from nla.models import (
    ActivationReconstructor,
    ActivationVerbalizer,
    ARConfig,
    AVConfig,
)
from nla.training.dataset import (
    LabeledPositionDataset,
    collate_labeled_positions,
)
from nla.training.fve import StratifiedFve

logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    activations_root: str
    labels_jsonl: str
    output_dir: str
    av_cfg: AVConfig = field(default_factory=AVConfig)
    ar_cfg: ARConfig = field(default_factory=ARConfig)
    seed: int = 0
    device: str = "cuda"

    held_out_fraction: float = 0.05
    batch_size: int = 4
    grad_accum_steps: int = 1
    grad_clip: float = 1.0
    learning_rate: float = 1e-4
    warmup_steps: int = 50
    total_steps: int = 1000
    weight_decay: float = 0.0

    av_weight: float = 1.0
    ar_weight: float = 1.0
    # Coefficient on AR's InfoNCE-style contrastive loss (see
    # ``ActivationReconstructor.forward_sft(return_nce=True)``).  Default 0 =
    # legacy MSE-only objective.  Set to e.g. 0.3 to penalize generic
    # descriptions that AR can reconstruct from any batch row.
    ar_contrastive_weight: float = 0.0

    # If True, multiply per-batch AV CE and AR MSE losses by the mean of
    # ``quality_weight`` over the batch (read from labels.jsonl by
    # ``LabeledPositionDataset``).  Backward compatible: labels without that
    # field default to weight 1.0.
    use_quality_weights: bool = False

    eval_every: int = 50
    save_every: int = 200
    log_every: int = 5

    gradient_checkpointing: bool = True

    # If True, draw training rows with a ``WeightedRandomSampler`` so per-batch
    # position_type frequencies approximate ``layer_spec.POSITION_MIX`` (40%
    # last_text / 40% image_patch / 20% anchor).  Off by default: the SFT split
    # is normally trained on whatever the labels file gave us.  Turn on when
    # the empirical histogram diverges materially from POSITION_MIX.
    balance_position_mix: bool = False

    # Drop labels whose ``description`` has fewer than this many markdown bullet
    # lines (lines starting with ``-``).  ``None`` disables the filter (legacy
    # behavior).  Used to cull degenerate captions before the split.
    min_bullet_lines: int | None = None

    # Closed-loop eval (h -> AV.generate -> AR -> ĥ) alongside the existing
    # teacher-forced eval.  Off by default for backwards compat and because
    # generation is expensive.  Temperatures of 0.0 are greedy; >0 are sampled.
    eval_closed_loop: bool = False
    closed_loop_temperatures: tuple[float, ...] = (0.0,)
    closed_loop_max_batches: int | None = None

    # ---- Scheduled sampling: feed AR its own AV-generated captions some of
    # the time (closes the SFT/eval distribution gap, makes the contrastive
    # NCE loss actually see template collapse and penalize it).
    #
    # Per-batch coin flip with prob ``p_av = ar_av_mix_max * ramp(step)``;
    # ramp is 0 until ``ar_av_mix_warmup_frac * total_steps`` then linear up
    # to 1 at total_steps. AV's own CE loss always uses gold tokens.
    #
    # V3 default raised from 0.0 to 0.3 (see docs/sft_plan/02_hyperparams.md
    # "V3 defaults"): the V2 postmortem showed the SFT/eval distribution gap
    # was a primary driver of the AR-reconstruction-shortcut failure mode.
    # CLI users can still pass ``--ar-av-mix-max 0`` to fall back to the
    # legacy gold-only objective.
    ar_av_mix_max: float = 0.3
    ar_av_mix_warmup_frac: float = 0.5
    ar_av_mix_max_new_tokens: int = 96
    ar_av_mix_do_sample: bool = False
    # For diagnostic logging only.
    ar_av_mix_log_text_every: int = 200

    # If labels are very few (e.g. smoke tests), allow the same item to repeat
    # without erroring.
    drop_last: bool = False

    # Optional caps so smoke runs finish quickly even if total_steps is high.
    max_train_items: int | None = None
    max_val_items: int | None = None

    # "episode" = hold out whole episodes (default; needed for the
    # memorization-vs-generalization metric).  "row" = legacy random row split.
    split_by: str = "episode"

    # When ``split_by == "episode"`` but the dump doesn't support it (no
    # ``episode_index`` or only one distinct episode), the dataset normally
    # logs a warning and falls back to a row split.  Set this to ``False`` for
    # paper / generalization runs to make that case raise ``RuntimeError``
    # instead of silently degrading.
    allow_episode_split_row_fallback: bool = True

    # Validate ``position_index < seq_len`` for every kept label at dataset
    # init time. Raises ``ValueError`` listing offenders if any row would
    # otherwise raise ``IndexError`` mid-training. Disable only for ablations
    # where you intentionally feed mismatched labels/activations.
    strict_position_check: bool = True

    # Hard-negative mining for AR's InfoNCE term. ``"none"`` (default) keeps
    # the legacy in-batch-only contrast. ``"same_episode"`` injects K_neg
    # captions sampled from the same episode but a different step, biasing
    # the negative set toward visually-similar-but-temporally-different
    # scenes. ``"same_position_type"`` samples K_neg from a different episode
    # whose label has the same position_type, biasing toward "same kind of
    # token, different scene". ``"topk_cosine"`` consumes a precomputed
    # JSONL of top-K activation-cosine neighbors (offline-mined by
    # ``scripts/training/mine_hard_negatives.py``); this is the
    # strongest form and what the V2 postmortem recommended to break
    # template collapse. Any non-none value is propagated to
    # ``LabeledPositionDataset`` and forces the collate fn to emit a
    # ``negative_descriptions`` key that ``ar.forward_sft`` consumes.
    ar_nce_hard_negative_source: Literal[
        "none", "same_episode", "same_position_type", "topk_cosine"
    ] = "none"
    ar_nce_hard_negatives_per_anchor: int = 4
    # Required when ``ar_nce_hard_negative_source == "topk_cosine"``; ignored
    # otherwise. Path to the mining JSONL produced by
    # ``scripts/training/mine_hard_negatives.py``.
    ar_nce_hard_negative_index_path: str | None = None

    # V4 image-patch read-time pooling. ``"pinned"`` (default) preserves V3
    # behaviour: for each ``image_patch`` row return the single-token
    # activation at ``entry.position_index`` exactly as the labeling pass
    # committed to. The non-pinned strategies pool over ALL valid image
    # patches in the example and return that pooled vector instead — for
    # ``image_patch`` rows only; ``last_text`` / ``anchor`` rows are
    # untouched. Per the V4 extraction A/B sweep
    # (``data/sft/libero_4suite_v3/v4_extraction_scorecard.json``)
    # ``"mean_pool_image"`` is the recommended V4 setting.
    image_patch_pooling: Literal[
        "pinned", "mean_pool_image", "strided_image", "center_image"
    ] = "pinned"
    image_patch_pooling_strided_k: int = 4

    # ---- Action-head consistency (Phase B scaffolding, see
    # ``src/nla/training/action_head_consistency.py`` and
    # ``docs/sft_plan/09_action_head_lora_phase1.md``).
    #
    # Default 0.0 keeps SFT byte-identical to V3/V4 baseline; the kernel
    # module is only imported when the weight is > 0.
    action_consistency_weight: float = 0.0
    # Run the consistency forward every N optimizer steps (1 = every step).
    action_consistency_every_n_steps: int = 8
    # Number of rows from the SFT batch that participate in the consistency
    # forward; the policy forward is the dominant per-step cost on a single
    # GPU, so we keep this small by default.
    action_consistency_max_microbatch: int = 1
    # When True, only feed rows whose ``position_type == "image_patch"``
    # into the consistency forward. This matches the steering placement
    # (image_patch_all) so the training-time loss is faithful to the
    # eval-time intervention.
    action_consistency_image_patch_only: bool = True
    # Path to a frozen GR00T checkpoint that gets used for the policy
    # forward.  Required when ``action_consistency_weight > 0``.
    action_consistency_policy_path: str | None = None
    # GR00T embodiment tag for the policy loader (e.g. "LIBERO_PANDA").
    action_consistency_embodiment_tag: str | None = None
    # JSON mapping ``{"<suite>": "<lerobot_dataset_root>"}`` (use the empty
    # string as the suite key for single-suite dumps without a prefix).
    action_consistency_dataset_roots: dict[str, str] | None = None
    # Where to cache the replay manifest JSONL (defaults to
    # ``<output_dir>/aux/replay_manifest.jsonl``).
    action_consistency_manifest_cache: str | None = None
    # Optional whitelist of suite names: when set, only manifest rows whose
    # suite is in this set participate in the consistency forward (all other
    # rows still go through the regular SFT objectives). ``None`` = no filter.
    action_consistency_suites: tuple[str, ...] | None = None


def _setup_outputs(out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "av": out_dir / "av",
        "ar": out_dir / "ar",
        "log": out_dir / "log",
        "metrics": out_dir / "metrics.jsonl",
        "config": out_dir / "config.json",
    }
    paths["av"].mkdir(exist_ok=True)
    paths["ar"].mkdir(exist_ok=True)
    paths["log"].mkdir(exist_ok=True)
    return paths


def _serialize_config(cfg: SFTConfig) -> dict[str, Any]:
    raw = asdict(cfg)
    # Replace tuples with lists for clean JSON.
    for sub in ("av_cfg", "ar_cfg"):
        if isinstance(raw.get(sub), dict):
            for k, v in list(raw[sub].items()):
                if isinstance(v, tuple):
                    raw[sub][k] = list(v)
    return raw


def _position_mix_sampler(labels, *, seed: int):
    """Build a ``WeightedRandomSampler`` that rebalances ``labels`` toward
    :data:`nla.layer_spec.POSITION_MIX`.

    Per-row weight ``w_i = POSITION_MIX[type_i] / (count[type_i] / N)``;
    rows whose ``position_type`` is not in ``POSITION_MIX`` get weight 0 (a
    safer choice than letting unrecognized types dominate).  Sampling is with
    replacement so a small minority class can fill its target share.
    """
    from torch.utils.data import WeightedRandomSampler

    n = len(labels)
    counts: dict[str, int] = {}
    for e in labels:
        counts[e.position_type] = counts.get(e.position_type, 0) + 1

    weights = torch.zeros(n, dtype=torch.double)
    for i, e in enumerate(labels):
        target = POSITION_MIX.get(e.position_type)
        if target is None:
            continue
        empirical = counts[e.position_type] / max(1, n)
        weights[i] = target / max(1e-12, empirical)

    if float(weights.sum()) <= 0.0:
        raise RuntimeError(
            "balance_position_mix=True but no label rows had a position_type "
            f"in POSITION_MIX ({sorted(POSITION_MIX)}); found types "
            f"{sorted(counts)}."
        )

    empirical_pct = {k: v / max(1, n) for k, v in counts.items()}
    logger.info(
        "position_mix rebalance: empirical=%s target=%s (excluded types: %s)",
        {k: round(v, 3) for k, v in empirical_pct.items()},
        {k: round(v, 3) for k, v in POSITION_MIX.items()},
        sorted(set(counts) - set(POSITION_MIX)) or "none",
    )

    g = torch.Generator()
    g.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=weights, num_samples=n, replacement=True, generator=g,
    )


def _make_dataloaders(cfg: SFTConfig):
    train_ds = LabeledPositionDataset(
        cfg.activations_root, cfg.labels_jsonl,
        seed=cfg.seed,
        strict_position_check=cfg.strict_position_check,
        held_out_fraction=cfg.held_out_fraction,
        held_out=False,
        max_items=cfg.max_train_items,
        split_by=cfg.split_by,
        allow_episode_split_row_fallback=cfg.allow_episode_split_row_fallback,
        min_bullet_lines=cfg.min_bullet_lines,
        hard_negative_source=cfg.ar_nce_hard_negative_source,
        hard_negatives_per_anchor=cfg.ar_nce_hard_negatives_per_anchor,
        hard_negative_index_path=cfg.ar_nce_hard_negative_index_path,
        image_patch_pooling=cfg.image_patch_pooling,
        image_patch_pooling_strided_k=cfg.image_patch_pooling_strided_k,
    )
    val_ds = LabeledPositionDataset(
        cfg.activations_root, cfg.labels_jsonl,
        seed=cfg.seed,
        strict_position_check=cfg.strict_position_check,
        held_out_fraction=cfg.held_out_fraction,
        held_out=True,
        max_items=cfg.max_val_items,
        split_by=cfg.split_by,
        allow_episode_split_row_fallback=cfg.allow_episode_split_row_fallback,
        min_bullet_lines=cfg.min_bullet_lines,
        image_patch_pooling=cfg.image_patch_pooling,
        image_patch_pooling_strided_k=cfg.image_patch_pooling_strided_k,
    )
    logger.info("Train labels: %d  Val labels: %d", len(train_ds), len(val_ds))

    if cfg.balance_position_mix:
        sampler = _position_mix_sampler(train_ds.labels, seed=cfg.seed)
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, sampler=sampler,
            num_workers=0, collate_fn=collate_labeled_positions,
            drop_last=cfg.drop_last,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=0, collate_fn=collate_labeled_positions,
            drop_last=cfg.drop_last,
        )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_labeled_positions,
        drop_last=False,
    )
    return train_loader, val_loader, train_ds, val_ds


def _build_models(cfg: SFTConfig):
    logger.info("Building AV (%s)", cfg.av_cfg.base_model)
    av = ActivationVerbalizer(cfg.av_cfg).to(cfg.device)
    logger.info("Building AR (%s, %d layers)", cfg.ar_cfg.base_model, cfg.ar_cfg.truncate_to_n_layers)
    # Share tokenizer with AV so we don't add the slot token to a second one.
    ar = ActivationReconstructor(cfg.ar_cfg, tokenizer=av.tokenizer).to(cfg.device)
    if cfg.gradient_checkpointing:
        for module in (av.base, ar.base):
            for fn in ("gradient_checkpointing_enable", "enable_input_require_grads"):
                if hasattr(module, fn):
                    try:
                        getattr(module, fn)()
                    except Exception as e:
                        logger.warning("Could not %s on %s: %s", fn, type(module).__name__, e)
    return av, ar


def _lr_schedule(step: int, cfg: SFTConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
    prog = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    return 0.5 * cfg.learning_rate * (1.0 + math.cos(math.pi * min(1.0, prog)))


def _ar_av_mix_p(step: int, cfg: SFTConfig) -> float:
    """Probability of swapping in AV-generated captions for AR's loss this step.

    Stays at 0 until ``ar_av_mix_warmup_frac * total_steps`` then ramps linearly
    up to ``ar_av_mix_max`` by ``total_steps``.  Clamped to [0, 1].
    """
    if cfg.ar_av_mix_max <= 0.0:
        return 0.0
    warmup_end = int(cfg.ar_av_mix_warmup_frac * cfg.total_steps)
    if step < warmup_end:
        return 0.0
    span = max(1, cfg.total_steps - warmup_end)
    prog = min(1.0, (step - warmup_end) / span)
    return max(0.0, min(1.0, prog * cfg.ar_av_mix_max))


def _write_jsonl_row(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _closed_loop_eval(
    av,
    ar,
    val_loader,
    device,
    *,
    temperature: float,
    max_batches: int | None,
) -> dict[str, float]:
    """Run h -> AV.generate -> AR -> ĥ and return stratified FVE/MSE/cosine.

    ``temperature == 0`` switches to greedy (``do_sample=False``).
    """
    fve_acc = StratifiedFve(group_name="position")
    n_seen = 0
    do_sample = temperature > 0.0
    temp_arg = float(temperature) if do_sample else 1.0
    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        acts = batch["activations"].to(device)
        gen_out = av.generate(
            activations=acts,
            position_types=batch["position_type"],
            do_sample=do_sample,
            temperature=temp_arg,
        )
        texts = gen_out["text"]
        pred_scaled = ar(texts, device=device)
        pred_unscaled = pred_scaled.detach().float() * ar.cfg.alpha
        fve_acc.update(acts.float(), pred_unscaled, batch["position_type"])
        n_seen += acts.shape[0]
    out = fve_acc.compute()
    out["_n_rows"] = float(n_seen)
    return out


@torch.no_grad()
def _evaluate(
    av,
    ar,
    val_loader,
    device,
    alpha: float,
    *,
    closed_loop_temperatures: tuple[float, ...] = (),
    closed_loop_max_batches: int | None = None,
) -> dict[str, float]:
    if val_loader is None or len(val_loader) == 0:
        return {}
    av.eval()
    ar.eval()
    ce_sum = 0.0
    ce_n = 0
    fve_acc = StratifiedFve(group_name="position")
    for batch in val_loader:
        acts = batch["activations"].to(device)
        out = av.forward_sft(
            activations=acts,
            position_types=batch["position_type"],
            target_texts=batch["description"],
        )
        ce_sum += float(out.loss.item()) * acts.shape[0]
        ce_n += acts.shape[0]
        pred_scaled = ar(batch["description"], device=device)
        pred_unscaled = pred_scaled.detach().float() * alpha
        fve_acc.update(acts.float(), pred_unscaled, batch["position_type"])
    metrics = fve_acc.compute()
    metrics["ce"] = ce_sum / max(1, ce_n)

    for temperature in closed_loop_temperatures:
        tag = "greedy" if temperature <= 0.0 else f"t{temperature:g}"
        cl = _closed_loop_eval(
            av, ar, val_loader, device,
            temperature=float(temperature),
            max_batches=closed_loop_max_batches,
        )
        for k, v in cl.items():
            metrics[f"closed_{tag}/{k}"] = v

    av.train()
    ar.train()
    return metrics


def _build_action_consistency_kernel(cfg: SFTConfig, ar, out_dir: Path):
    """Build the action-head consistency kernel when enabled.

    Returns ``None`` when the consistency loss is disabled so the train loop
    can skip every related branch in O(1). Imports the kernel module lazily so
    that ``run_sft`` doesn't pull in GR00T-adjacent code when the auxiliary
    objective is off.
    """
    if cfg.action_consistency_weight <= 0.0:
        return None
    if not cfg.action_consistency_policy_path:
        raise ValueError(
            "action_consistency_weight > 0 requires --action-consistency-policy-path "
            "to point at a frozen GR00T checkpoint."
        )
    if not cfg.action_consistency_embodiment_tag:
        raise ValueError(
            "action_consistency_weight > 0 requires "
            "--action-consistency-embodiment-tag (e.g. LIBERO_PANDA); the policy "
            "loader needs it to resolve DATA_CONFIG_MAP."
        )
    if not cfg.action_consistency_dataset_roots:
        raise ValueError(
            "action_consistency_weight > 0 requires --action-consistency-dataset-roots "
            "(JSON mapping suite -> LeRobot dataset root)."
        )

    from nla.training.action_head_consistency import (
        ActionConsistencyConfig,
        ActionConsistencyKernel,
        make_lerobot_obs_builder,
    )
    from nla.training.replay_manifest import build_replay_manifest

    cache = (
        Path(cfg.action_consistency_manifest_cache)
        if cfg.action_consistency_manifest_cache
        else out_dir / "aux" / "replay_manifest.jsonl"
    )
    roots: dict[str | None, str] = {}
    for k, v in (cfg.action_consistency_dataset_roots or {}).items():
        # Treat the empty string as "no suite prefix" (single-suite dumps).
        roots[None if not k else str(k)] = str(v)

    # Optionally restrict the manifest to a whitelist of suites. This lets us
    # run consistency on, e.g., libero_goal only while the other three suites
    # still flow through the regular SFT objectives.
    suite_whitelist: set[str] | None = None
    if cfg.action_consistency_suites:
        suite_whitelist = {str(s) for s in cfg.action_consistency_suites}
        roots = {k: v for k, v in roots.items() if k in suite_whitelist}
        if not roots:
            raise ValueError(
                "action_consistency_suites filtered the dataset_roots map to "
                f"empty (whitelist={sorted(suite_whitelist)}, available="
                f"{sorted(cfg.action_consistency_dataset_roots or {})}). "
                "Either drop the filter or include at least one matching suite."
            )

    manifest = build_replay_manifest(
        cfg.activations_root, roots, cache_path=cache,
    )

    if not manifest:
        raise RuntimeError(
            "Replay manifest is empty after building from "
            f"{cfg.activations_root} with suites={sorted(roots)}. The "
            "consistency kernel will have no rows to operate on; either "
            "broaden the suite filter or check that the activation example_ids "
            "are parseable."
        )

    # Defer GR00T import to the policy loader so we can fail loudly only if
    # the kernel actually fires. We mirror the construction used elsewhere
    # in the codebase (e.g. ``scripts/eval/run_gr00t_server_nla_steer.py``),
    # which lets ``Gr00tPolicy`` auto-load modality configs from the
    # checkpoint -- DATA_CONFIG_MAP doesn't exist on this gr00t version.
    def _policy_loader():
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        policy = Gr00tPolicy(
            embodiment_tag=cfg.action_consistency_embodiment_tag,
            model_path=cfg.action_consistency_policy_path,
            device=cfg.device,
        )
        policy.model.eval()
        return policy

    # The LeRobot obs_builder needs ``policy.modality_configs`` to construct
    # its loaders, which only exists after ``_policy_loader`` has run. The
    # kernel resolves the factory inside ``ensure_loaded()`` for that reason.
    roots_str_only: dict[str, str] = {
        k: v for k, v in roots.items() if k is not None
    }

    def _obs_builder_factory(policy):
        return make_lerobot_obs_builder(
            policy,
            roots_str_only,
            cfg.action_consistency_embodiment_tag,
        )

    return ActionConsistencyKernel(
        ActionConsistencyConfig(
            weight=float(cfg.action_consistency_weight),
            every_n_steps=int(cfg.action_consistency_every_n_steps),
            max_microbatch_per_step=int(cfg.action_consistency_max_microbatch),
            image_patch_rows_only=bool(cfg.action_consistency_image_patch_only),
        ),
        manifest=manifest,
        policy_loader=_policy_loader,
        obs_builder_factory=_obs_builder_factory,
        ar_module=ar,
        device=cfg.device,
    )


def run_sft(cfg: SFTConfig) -> dict[str, Any]:
    out_dir = Path(cfg.output_dir)
    paths = _setup_outputs(out_dir)
    paths["config"].write_text(json.dumps(_serialize_config(cfg), indent=2))

    torch.manual_seed(cfg.seed)
    train_loader, val_loader, train_ds, val_ds = _make_dataloaders(cfg)
    if len(train_ds) == 0:
        raise RuntimeError("No labeled training examples; check labels.jsonl.")

    av, ar = _build_models(cfg)
    consistency_kernel = _build_action_consistency_kernel(cfg, ar, out_dir)
    if consistency_kernel is not None:
        logger.info(
            "Action-head consistency ENABLED: weight=%.3f cadence=every_%d_steps "
            "microbatch=%d image_patch_only=%s",
            cfg.action_consistency_weight,
            cfg.action_consistency_every_n_steps,
            cfg.action_consistency_max_microbatch,
            cfg.action_consistency_image_patch_only,
        )

    trainable_params = [
        p for p in list(av.parameters()) + list(ar.parameters()) if p.requires_grad
    ]
    n_trainable = sum(p.numel() for p in trainable_params)
    logger.info("Trainable params: %d (~%.2fM)", n_trainable, n_trainable / 1e6)

    optim = torch.optim.AdamW(
        trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )

    # Optional tensorboard.
    tb = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(str(paths["log"]))
    except Exception:
        logger.info("tensorboard unavailable; skipping TB logging.")

    av.train()
    ar.train()

    step = 0
    accum_count = 0
    optim.zero_grad(set_to_none=True)
    start = time.time()
    final_metrics: dict[str, float] = {}

    while step < cfg.total_steps:
        for batch in train_loader:
            if step >= cfg.total_steps:
                break
            for g in optim.param_groups:
                g["lr"] = _lr_schedule(step, cfg)

            acts = batch["activations"].to(cfg.device, non_blocking=True)
            descs = batch["description"]
            ptypes = batch["position_type"]

            av_out = av.forward_sft(
                activations=acts, position_types=ptypes, target_texts=descs,
            )
            ce = av_out.loss

            # Scheduled sampling: with probability p_av use AV's own generated
            # text as AR's input (and as the contrastive batch).  AV's CE loss
            # above is unchanged -- it always trains on gold.
            p_av = _ar_av_mix_p(step, cfg)
            ar_input_src = "gold"
            ar_input_text = descs
            if p_av > 0.0 and torch.rand((), device="cpu").item() < p_av:
                av.eval()
                with torch.no_grad():
                    gen_out = av.generate(
                        activations=acts,
                        position_types=ptypes,
                        max_new_tokens=cfg.ar_av_mix_max_new_tokens,
                        do_sample=bool(cfg.ar_av_mix_do_sample),
                        temperature=0.7 if cfg.ar_av_mix_do_sample else 1.0,
                    )
                av.train()
                ar_input_text = [t.strip() or "(empty)" for t in gen_out["text"]]
                ar_input_src = "av"
                if (
                    cfg.ar_av_mix_log_text_every > 0
                    and step % cfg.ar_av_mix_log_text_every == 0
                ):
                    sample = ar_input_text[0][:240].replace("\n", " ")
                    logger.info("[step %d] p_av=%.2f mix sample: %s", step, p_av, sample)

            if cfg.ar_contrastive_weight > 0.0:
                # Hard negatives are mined w.r.t. the anchor's activation
                # row, so they remain valid "describe-a-different-scene"
                # negatives whether the anchor caption is gold or
                # AV-generated. Pure MSE-only training
                # (ar_contrastive_weight == 0) ignores them by design.
                neg_descs = batch.get("negative_descriptions")
                ar_mse, ar_nce, _pred_scaled = ar.forward_sft(
                    ar_input_text, acts, return_nce=True,
                    negative_explanations=neg_descs,
                )
                ar_term = ar_mse + cfg.ar_contrastive_weight * ar_nce
            else:
                ar_mse, _pred_scaled = ar.forward_sft(ar_input_text, acts)
                ar_nce = torch.zeros((), device=acts.device)
                ar_term = ar_mse

            if cfg.use_quality_weights and "quality_weight" in batch:
                qw = batch["quality_weight"].to(acts.device).float().mean().clamp(min=0.0, max=1.0)
            else:
                qw = torch.ones((), device=acts.device)

            consistency_loss_t = torch.zeros((), device=acts.device)
            consistency_diag = None
            if (
                consistency_kernel is not None
                and step % cfg.action_consistency_every_n_steps == 0
            ):
                consistency_loss_t, consistency_diag = consistency_kernel.consistency_loss(
                    descriptions=descs,
                    example_ids=batch["example_id"],
                    position_types=ptypes,
                )

            loss = qw * (cfg.av_weight * ce + cfg.ar_weight * ar_term)
            if consistency_kernel is not None and consistency_diag is not None and consistency_diag.n_rows > 0:
                loss = loss + cfg.action_consistency_weight * consistency_loss_t
            (loss / max(1, cfg.grad_accum_steps)).backward()
            accum_count += 1

            if accum_count >= cfg.grad_accum_steps:
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
                optim.step()
                optim.zero_grad(set_to_none=True)
                accum_count = 0

            if step % cfg.log_every == 0:
                row = {
                    "step": step,
                    "phase": "train",
                    "ce": float(ce.detach().item()),
                    "ar_mse": float(ar_mse.detach().item()),
                    "ar_nce": float(ar_nce.detach().item()),
                    "qw_mean": float(qw.detach().item()),
                    "loss": float(loss.detach().item()),
                    "lr": optim.param_groups[0]["lr"],
                    "elapsed_s": time.time() - start,
                    "p_av": float(p_av),
                    "ar_mix_used": 1 if ar_input_src == "av" else 0,
                }
                if consistency_diag is not None:
                    row["action_consistency_loss"] = float(consistency_diag.loss)
                    row["action_consistency_n_rows"] = int(consistency_diag.n_rows)
                    row["action_consistency_delta_norm"] = float(consistency_diag.delta_action_norm)
                    row["action_consistency_cache_hits"] = int(consistency_diag.baseline_cache_hits)
                    row["action_consistency_cache_misses"] = int(consistency_diag.baseline_cache_misses)
                _write_jsonl_row(paths["metrics"], row)
                if tb is not None:
                    tb.add_scalar("train/ce", row["ce"], step)
                    tb.add_scalar("train/ar_mse", row["ar_mse"], step)
                    tb.add_scalar("train/ar_nce", row["ar_nce"], step)
                    tb.add_scalar("train/qw_mean", row["qw_mean"], step)
                    tb.add_scalar("train/loss", row["loss"], step)
                    tb.add_scalar("train/lr", row["lr"], step)
                    tb.add_scalar("train/p_av", row["p_av"], step)
                    tb.add_scalar("train/ar_mix_used", row["ar_mix_used"], step)
                    if consistency_diag is not None and consistency_diag.n_rows > 0:
                        tb.add_scalar("train/action_consistency_loss", row["action_consistency_loss"], step)
                        tb.add_scalar(
                            "train/action_consistency_delta_norm",
                            row["action_consistency_delta_norm"],
                            step,
                        )

            if step > 0 and step % cfg.eval_every == 0:
                metrics = _evaluate(
                    av, ar, val_loader, cfg.device, cfg.av_cfg.alpha,
                    closed_loop_temperatures=(
                        cfg.closed_loop_temperatures if cfg.eval_closed_loop else ()
                    ),
                    closed_loop_max_batches=cfg.closed_loop_max_batches,
                )
                if metrics:
                    row = {"step": step, "phase": "val", **metrics, "elapsed_s": time.time() - start}
                    _write_jsonl_row(paths["metrics"], row)
                    logger.info("[step %d] val %s", step, "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
                    if tb is not None:
                        for k, v in metrics.items():
                            tb.add_scalar(f"val/{k}", v, step)
                    final_metrics = metrics

            if step > 0 and step % cfg.save_every == 0:
                av.save(str(paths["av"]))
                ar.save(str(paths["ar"]))

            step += 1

    # Final eval + save.
    metrics = _evaluate(
        av, ar, val_loader, cfg.device, cfg.av_cfg.alpha,
        closed_loop_temperatures=(
            cfg.closed_loop_temperatures if cfg.eval_closed_loop else ()
        ),
        closed_loop_max_batches=cfg.closed_loop_max_batches,
    )
    if metrics:
        row = {"step": step, "phase": "final", **metrics, "elapsed_s": time.time() - start}
        _write_jsonl_row(paths["metrics"], row)
        logger.info("[final] val %s", "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        final_metrics = metrics

    av.save(str(paths["av"]))
    ar.save(str(paths["ar"]))
    if tb is not None:
        tb.close()
    return {"steps": step, "metrics": final_metrics, "out_dir": str(out_dir)}
