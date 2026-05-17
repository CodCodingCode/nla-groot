#!/usr/bin/env python
"""Intervention leverage sweep over backbone steer slots (open-loop Δaction).

For one ``(trajectory, step)``, this script applies the existing
``attach_backbone_steer`` hook over a grid of :class:`nla.steering.SteerSpec`
placements (``last_text``, ``anchor``, ``image_patch`` × seeds,
``image_patch_all``, ``fixed`` × token range), optionally with matched-norm
Gaussian **null controls**, and writes a ranked JSONL/CSV of ``|Δaction|`` per
condition. The goal is to discover **which extracted slot identities** give
high steering leverage on the action head — not who activates most, but who
changes behavior most under matched perturbations.

This is open-loop (single observation, no sim rollout). It uses the same
``policy.get_action`` definition of "action" as
``scripts/eval/nla_steer_groot_action.py``.

Example (LIBERO)::

    PYTHONPATH=src python scripts/eval/nla_steer_leverage_sweep.py \\
        --model-path     checkpoints/GR00T-N1.7-LIBERO/libero_goal \\
        --dataset-path   third_party/Isaac-GR00T/examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot \\
        --embodiment-tag LIBERO_PANDA \\
        --ar-dir         data/sft/libero_goal_pilot_v3/ar \\
        --traj-id 0 --step 0 \\
        --text-file      steer_bullets.txt \\
        --placements     last_text,anchor,image_patch \\
        --image-patch-seeds 0,1,2,3,4 \\
        --fixed-token-range 0::16 \\
        --null-samples   4 \\
        --out-jsonl      data/sft/libero_goal_pilot_v3/intervention_leverage.jsonl \\
        --out-csv        data/sft/libero_goal_pilot_v3/intervention_leverage.csv

``--embodiment-tag`` accepts either the GR00T enum name (e.g. ``LIBERO_PANDA``,
``BRIDGE_V2``) or its lower-cased value (``libero_sim``, ``bridge_v2``).
LIBERO is the supported target; non-LIBERO embodiments are passed through
unchanged but not exercised in our eval suite.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


logger = logging.getLogger("nla.steer_sweep")


def _imports():
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    from nla.extraction._compat import apply_all as apply_groot_compat

    apply_groot_compat()

    return dict(
        LeRobotEpisodeLoader=LeRobotEpisodeLoader,
        extract_step_data=extract_step_data,
        EmbodimentTag=EmbodimentTag,
        Gr00tPolicy=Gr00tPolicy,
    )


def _load_steer_text(args: argparse.Namespace) -> str:
    if args.text and args.text_file:
        raise SystemExit("Use only one of --text or --text-file")
    if args.text_file:
        return Path(args.text_file).read_text()
    if args.text:
        return args.text
    raise SystemExit("Provide --text or --text-file with your AR bullet description.")


def _parse_int_list(spec: str | None) -> list[int]:
    if not spec:
        return []
    return [int(x) for x in spec.split(",") if x.strip()]


def _parse_float_list(spec: str) -> list[float]:
    return [float(x) for x in spec.split(",") if x.strip()]


def _parse_range_spec(spec: str | None) -> tuple[int, int | None, int] | None:
    """Parse ``start:end:stride`` (``end`` may be empty → ``None`` meaning T)."""
    if not spec:
        return None
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise SystemExit(f"--fixed-token-range expects start:end[:stride], got {spec!r}")
    start = int(parts[0]) if parts[0].strip() else 0
    end = int(parts[1]) if parts[1].strip() else None
    stride = int(parts[2]) if len(parts) == 3 and parts[2].strip() else 1
    if stride <= 0:
        raise SystemExit("--fixed-token-range stride must be >= 1")
    return start, end, stride


@dataclass(frozen=True)
class Condition:
    label: str
    placement: str
    blend: float
    fixed_token_index: int | None
    image_patch_seed: int | None


class _ProbeHook:
    """Read-only forward hook: capture mask shapes during the baseline forward."""

    def __init__(self) -> None:
        self.attention_mask: torch.Tensor | None = None
        self.image_mask: torch.Tensor | None = None
        self.feature_shape: tuple[int, ...] | None = None

    def __call__(self, module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        del module, inputs
        attn = output["backbone_attention_mask"]
        img = output["image_mask"]
        feats = output["backbone_features"]
        self.attention_mask = attn.detach().to(device="cpu", copy=True)
        self.image_mask = img.detach().to(device="cpu", copy=True)
        self.feature_shape = tuple(feats.shape)


def _build_conditions(
    *,
    placements: list[str],
    blends: list[float],
    image_patch_seeds: list[int],
    fixed_token_indices: list[int],
) -> list[Condition]:
    valid = {"last_text", "anchor", "image_patch", "image_patch_all", "fixed"}
    conds: list[Condition] = []
    for blend in blends:
        for placement in placements:
            if placement not in valid:
                raise SystemExit(f"Unknown placement {placement!r}; valid: {sorted(valid)}")
            if placement == "image_patch":
                if not image_patch_seeds:
                    raise SystemExit(
                        "placement 'image_patch' needs --image-patch-seeds (e.g. 0,1,2)"
                    )
                for seed in image_patch_seeds:
                    conds.append(Condition(
                        label=f"image_patch[seed={seed}]@blend={blend}",
                        placement="image_patch",
                        blend=blend,
                        fixed_token_index=None,
                        image_patch_seed=int(seed),
                    ))
            elif placement == "fixed":
                if not fixed_token_indices:
                    raise SystemExit(
                        "placement 'fixed' needs --fixed-token-indices or --fixed-token-range"
                    )
                for t in fixed_token_indices:
                    conds.append(Condition(
                        label=f"fixed[t={t}]@blend={blend}",
                        placement="fixed",
                        blend=blend,
                        fixed_token_index=int(t),
                        image_patch_seed=None,
                    ))
            else:
                conds.append(Condition(
                    label=f"{placement}@blend={blend}",
                    placement=placement,
                    blend=blend,
                    fixed_token_index=None,
                    image_patch_seed=None,
                ))
    return conds


def _matched_null_vec(real_vec_cpu: torch.Tensor, seed: int) -> torch.Tensor:
    """Gaussian draw rescaled to match ``||real_vec||_2`` (float32, CPU)."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    z = torch.randn(real_vec_cpu.shape, generator=gen, dtype=torch.float32)
    target = float(torch.linalg.norm(real_vec_cpu.float()))
    z_norm = float(torch.linalg.norm(z))
    if z_norm < 1e-12:
        return z
    return z * (target / z_norm)


def _resolve_indices(probe: _ProbeHook, spec: Any) -> list[int]:
    """Read-only mirror of what the steer hook will pick for ``spec``."""
    from nla.steering.backbone_steer import resolve_steer_indices

    if probe.attention_mask is None or probe.image_mask is None:
        return []
    try:
        return resolve_steer_indices(probe.attention_mask, probe.image_mask, spec)
    except Exception as e:
        logger.warning("resolve_steer_indices failed for spec=%r: %s", spec, e)
        return []


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model-path", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--embodiment-tag", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ar-dir", required=True, help="SFT ``ar/`` checkpoint directory")
    p.add_argument("--traj-id", type=int, default=0)
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--video-backend", default="torchcodec",
                   choices=["decord", "torchvision_av", "torchcodec"])
    p.add_argument("--text", default=None, help="Inline bullet text for AR")
    p.add_argument("--text-file", default=None, help="UTF-8 file with bullet text")
    p.add_argument(
        "--placements",
        default="last_text,anchor",
        help="Comma list: last_text, anchor, image_patch, image_patch_all, fixed",
    )
    p.add_argument(
        "--blends",
        default="1.0",
        help="Comma list of blend factors (default just 1.0 = hard replace)",
    )
    p.add_argument(
        "--image-patch-seeds",
        default="",
        help="Comma list of seeds (used when placements includes 'image_patch')",
    )
    p.add_argument(
        "--fixed-token-range",
        default=None,
        help="``start:end:stride`` for placement='fixed' (end empty means T)",
    )
    p.add_argument(
        "--fixed-token-indices",
        default=None,
        help="Comma list of explicit token indices for placement='fixed' (overrides range)",
    )
    p.add_argument(
        "--null-samples",
        type=int,
        default=0,
        help="Matched-norm Gaussian draws per condition for null control (0 disables)",
    )
    p.add_argument("--null-seed", type=int, default=0)
    p.add_argument(
        "--sort-by",
        default="effect",
        choices=["effect", "delta_vs_null", "label", "order"],
        help="JSONL/CSV row ordering",
    )
    p.add_argument("--out-jsonl", required=True, help="Write one JSON object per condition")
    p.add_argument("--out-csv", default=None, help="Optional flat CSV for spreadsheet sorting")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    from nla.steering import (
        SteerSpec,
        action_stats,
        ar_text_to_backbone_vec,
        attach_backbone_steer,
        policy_get_action,
    )
    from nla.steering.groot_obs import build_observation_for_step
    from nla.training.checkpoint import load_ar_from_sft

    placements = [s.strip() for s in args.placements.split(",") if s.strip()]
    blends = _parse_float_list(args.blends)
    image_patch_seeds = _parse_int_list(args.image_patch_seeds)

    steer_text = _load_steer_text(args)
    mods = _imports()
    embodiment_tag = mods["EmbodimentTag"].resolve(args.embodiment_tag)

    logger.info("Loading policy…")
    policy = mods["Gr00tPolicy"](
        embodiment_tag=embodiment_tag,
        model_path=args.model_path,
        device=args.device,
    )

    modality_configs = deepcopy(policy.modality_configs)
    modality_configs.pop("action", None)

    loader = mods["LeRobotEpisodeLoader"](
        dataset_path=args.dataset_path,
        modality_configs=policy.modality_configs,
        video_backend=args.video_backend,
    )
    language_keys = list(policy.modality_configs["language"].modality_keys)
    traj = loader[args.traj_id]

    obs = build_observation_for_step(
        traj,
        args.step,
        modality_configs,
        embodiment_tag,
        language_keys,
        mods["extract_step_data"],
    )

    logger.info("Loading AR from %s …", args.ar_dir)
    ar = load_ar_from_sft(Path(args.ar_dir), device=args.device, freeze=True)
    steer_vec = ar_text_to_backbone_vec(ar, steer_text).to(args.device)
    steer_vec_cpu = steer_vec.detach().float().cpu().contiguous()
    steer_norm = float(torch.linalg.norm(steer_vec_cpu))
    logger.info("steer_vec L2 norm = %.4f (dim=%d)", steer_norm, steer_vec_cpu.shape[0])

    if not hasattr(policy.model, "backbone"):
        raise RuntimeError("policy.model has no .backbone; GR00T layout changed?")
    backbone = policy.model.backbone
    policy.model.eval()

    logger.info("Probing baseline forward to resolve sequence length T…")
    probe = _ProbeHook()
    handle = backbone.register_forward_hook(probe)
    try:
        with torch.inference_mode():
            base_action = policy_get_action(policy, obs)
    finally:
        handle.remove()

    if probe.attention_mask is None or probe.feature_shape is None:
        raise RuntimeError(
            "Probe hook did not capture backbone output. GR00T layout changed?"
        )
    T = int(probe.attention_mask.shape[-1])
    logger.info("Backbone sequence length T = %d (feature shape %s)", T, probe.feature_shape)

    fixed_token_indices: list[int] = []
    if args.fixed_token_indices:
        for t in _parse_int_list(args.fixed_token_indices):
            if 0 <= t < T:
                fixed_token_indices.append(t)
            else:
                logger.warning(
                    "dropping --fixed-token-indices entry %d (out of [0, %d))", t, T
                )
    elif args.fixed_token_range and "fixed" in placements:
        rng_spec = _parse_range_spec(args.fixed_token_range)
        if rng_spec is not None:
            start, end, stride = rng_spec
            if end is None or end > T:
                end = T
            fixed_token_indices = list(range(max(0, start), end, stride))

    if "fixed" in placements and not fixed_token_indices:
        raise SystemExit(
            "placement 'fixed' requested but no usable token indices "
            "(provide --fixed-token-indices or --fixed-token-range)"
        )

    conditions = _build_conditions(
        placements=placements,
        blends=blends,
        image_patch_seeds=image_patch_seeds,
        fixed_token_indices=fixed_token_indices,
    )
    total_forwards = len(conditions) * (1 + max(0, int(args.null_samples)))
    logger.info(
        "Sweep: %d conditions × (1 effect + %d null) = %d steered forwards",
        len(conditions), args.null_samples, total_forwards,
    )

    rows: list[dict[str, Any]] = []
    null_seed_counter = int(args.null_seed)

    for ci, cond in enumerate(conditions):
        spec = SteerSpec(
            placement=cond.placement,  # type: ignore[arg-type]
            blend=cond.blend,
            fixed_token_index=cond.fixed_token_index,
            image_patch_seed=int(cond.image_patch_seed or 0),
        )
        resolved_idxs = _resolve_indices(probe, spec)

        t0 = time.time()
        with torch.inference_mode():
            with attach_backbone_steer(backbone, steer_vec, spec):
                steered = policy_get_action(policy, obs)
        effect = action_stats(base_action, steered)
        effect_elapsed = time.time() - t0

        null_global_max: list[float] = []
        null_elapsed_total = 0.0
        for _ in range(int(args.null_samples)):
            null_vec = _matched_null_vec(steer_vec_cpu, null_seed_counter)
            null_seed_counter += 1
            tn = time.time()
            with torch.inference_mode():
                with attach_backbone_steer(backbone, null_vec.to(args.device), spec):
                    steered_null = policy_get_action(policy, obs)
            null_stats = action_stats(base_action, steered_null)
            null_global_max.append(float(null_stats["global_max_abs"]))
            null_elapsed_total += time.time() - tn

        null_median = float(np.median(null_global_max)) if null_global_max else 0.0
        null_p95 = (
            float(np.percentile(null_global_max, 95.0)) if null_global_max else 0.0
        )
        delta_vs_null_median = float(effect["global_max_abs"] - null_median)

        row = {
            "condition_idx": ci,
            "label": cond.label,
            "placement": cond.placement,
            "blend": cond.blend,
            "fixed_token_index": cond.fixed_token_index,
            "image_patch_seed": cond.image_patch_seed,
            "resolved_indices": resolved_idxs,
            "num_indices": len(resolved_idxs),
            "T": T,
            "steer_vec_l2": steer_norm,
            "effect": effect,
            "null_count": len(null_global_max),
            "null_global_max_abs": null_global_max,
            "null_global_max_abs_median": null_median,
            "null_global_max_abs_p95": null_p95,
            "delta_vs_null_median": delta_vs_null_median,
            "elapsed_effect_s": effect_elapsed,
            "elapsed_null_s": null_elapsed_total,
        }
        rows.append(row)
        logger.info(
            "[%3d/%d] %-40s |Δa|=%.6f  null_med=%.6f  Δvs.null=%+.6f  (%.2fs+%.2fs)",
            ci + 1, len(conditions), cond.label,
            effect["global_max_abs"], null_median, delta_vs_null_median,
            effect_elapsed, null_elapsed_total,
        )

    if args.sort_by == "effect":
        rows.sort(key=lambda r: r["effect"]["global_max_abs"], reverse=True)
    elif args.sort_by == "delta_vs_null":
        rows.sort(key=lambda r: r["delta_vs_null_median"], reverse=True)
    elif args.sort_by == "label":
        rows.sort(key=lambda r: r["label"])

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    logger.info("Wrote %s (%d rows)", out_jsonl, len(rows))

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "condition_idx", "label", "placement", "blend",
            "fixed_token_index", "image_patch_seed",
            "num_indices", "T", "steer_vec_l2",
            "effect_global_max_abs",
            "null_count", "null_global_max_abs_median", "null_global_max_abs_p95",
            "delta_vs_null_median", "elapsed_effect_s", "elapsed_null_s",
        ]
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({
                    "condition_idx": r["condition_idx"],
                    "label": r["label"],
                    "placement": r["placement"],
                    "blend": r["blend"],
                    "fixed_token_index": r["fixed_token_index"],
                    "image_patch_seed": r["image_patch_seed"],
                    "num_indices": r["num_indices"],
                    "T": r["T"],
                    "steer_vec_l2": r["steer_vec_l2"],
                    "effect_global_max_abs": r["effect"]["global_max_abs"],
                    "null_count": r["null_count"],
                    "null_global_max_abs_median": r["null_global_max_abs_median"],
                    "null_global_max_abs_p95": r["null_global_max_abs_p95"],
                    "delta_vs_null_median": r["delta_vs_null_median"],
                    "elapsed_effect_s": r["elapsed_effect_s"],
                    "elapsed_null_s": r["elapsed_null_s"],
                })
        logger.info("Wrote %s", out_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
