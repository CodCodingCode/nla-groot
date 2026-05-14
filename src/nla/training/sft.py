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
from typing import Any

import torch
from torch.utils.data import DataLoader

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


def _make_dataloaders(cfg: SFTConfig):
    train_ds = LabeledPositionDataset(
        cfg.activations_root, cfg.labels_jsonl,
        seed=cfg.seed,
        held_out_fraction=cfg.held_out_fraction,
        held_out=False,
        max_items=cfg.max_train_items,
        split_by=cfg.split_by,
        allow_episode_split_row_fallback=cfg.allow_episode_split_row_fallback,
    )
    val_ds = LabeledPositionDataset(
        cfg.activations_root, cfg.labels_jsonl,
        seed=cfg.seed,
        held_out_fraction=cfg.held_out_fraction,
        held_out=True,
        max_items=cfg.max_val_items,
        split_by=cfg.split_by,
        allow_episode_split_row_fallback=cfg.allow_episode_split_row_fallback,
    )
    logger.info("Train labels: %d  Val labels: %d", len(train_ds), len(val_ds))
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


def _write_jsonl_row(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


@torch.no_grad()
def _evaluate(av, ar, val_loader, device, alpha: float) -> dict[str, float]:
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
    av.train()
    ar.train()
    return metrics


def run_sft(cfg: SFTConfig) -> dict[str, Any]:
    out_dir = Path(cfg.output_dir)
    paths = _setup_outputs(out_dir)
    paths["config"].write_text(json.dumps(_serialize_config(cfg), indent=2))

    torch.manual_seed(cfg.seed)
    train_loader, val_loader, train_ds, val_ds = _make_dataloaders(cfg)
    if len(train_ds) == 0:
        raise RuntimeError("No labeled training examples; check labels.jsonl.")

    av, ar = _build_models(cfg)

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

            if cfg.ar_contrastive_weight > 0.0:
                ar_mse, ar_nce, _pred_scaled = ar.forward_sft(descs, acts, return_nce=True)
                ar_term = ar_mse + cfg.ar_contrastive_weight * ar_nce
            else:
                ar_mse, _pred_scaled = ar.forward_sft(descs, acts)
                ar_nce = torch.zeros((), device=acts.device)
                ar_term = ar_mse

            if cfg.use_quality_weights and "quality_weight" in batch:
                qw = batch["quality_weight"].to(acts.device).float().mean().clamp(min=0.0, max=1.0)
            else:
                qw = torch.ones((), device=acts.device)

            loss = qw * (cfg.av_weight * ce + cfg.ar_weight * ar_term)
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
                }
                _write_jsonl_row(paths["metrics"], row)
                if tb is not None:
                    tb.add_scalar("train/ce", row["ce"], step)
                    tb.add_scalar("train/ar_mse", row["ar_mse"], step)
                    tb.add_scalar("train/ar_nce", row["ar_nce"], step)
                    tb.add_scalar("train/qw_mean", row["qw_mean"], step)
                    tb.add_scalar("train/loss", row["loss"], step)
                    tb.add_scalar("train/lr", row["lr"], step)

            if step > 0 and step % cfg.eval_every == 0:
                metrics = _evaluate(av, ar, val_loader, cfg.device, cfg.av_cfg.alpha)
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
    metrics = _evaluate(av, ar, val_loader, cfg.device, cfg.av_cfg.alpha)
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
