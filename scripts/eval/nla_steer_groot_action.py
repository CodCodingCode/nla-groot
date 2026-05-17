#!/usr/bin/env python
"""Causal probe: steer GR00T with **AR(text) → backbone vector** at a token site.

This is the cheap “language → ĥ → patch backbone → read Δaction” loop. It is
**not** a guarantee of semantically correct behavior (your AR may not be
faithful out-of-distribution); it is infrastructure to **measure** how much
action latents move when you inject an NLA reconstruction at
``backbone_features`` (same hook space as extraction).

Requires the Isaac-GR00T stack (Python 3.10 venv, ``pip install -e`` the
vendored GR00T repo) exactly like ``scripts/extraction/run_extract.py``.

Example (LIBERO)::

    PYTHONPATH=src python scripts/eval/nla_steer_groot_action.py \\
        --model-path     checkpoints/GR00T-N1.7-LIBERO/libero_goal \\
        --dataset-path   third_party/Isaac-GR00T/examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot \\
        --embodiment-tag LIBERO_PANDA \\
        --ar-dir         data/sft/libero_goal_pilot_v3/ar \\
        --traj-id        0 --step 0 \\
        --placement      image_patch \\
        --text-file      my_steer_bullets.txt

``--embodiment-tag`` accepts either the GR00T enum name (e.g. ``LIBERO_PANDA``,
``BRIDGE_V2``) or its lower-cased value (``libero_sim``, ``bridge_v2``).
LIBERO is the supported target; non-LIBERO embodiments are passed through
unchanged but not exercised in our eval suite. ``--text`` / ``--text-file``
should use the same bullet
style as your labeling pipeline (AR was trained on
``Summary of the following text: <text>…</text>``).
"""

from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path

import torch


logger = logging.getLogger("nla.steer_groot")


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


def _preflight_processor_access(model_path: str) -> None:
    import os

    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return
    try:
        from huggingface_hub import HfApi  # type: ignore

        HfApi().model_info("nvidia/Cosmos-Reason2-2B")
    except Exception as e:  # pragma: no cover
        logger.warning(
            "Could not verify Cosmos-Reason2 hub access (%s). Set HF_TOKEN if download fails.",
            type(e).__name__,
        )


def _load_steer_text(args: argparse.Namespace) -> str:
    if args.text and args.text_file:
        raise SystemExit("Use only one of --text or --text-file")
    if args.text_file:
        return Path(args.text_file).read_text()
    if args.text:
        return args.text
    raise SystemExit("Provide --text or --text-file with your AR bullet description.")


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
        "--placement",
        default="image_patch",
        choices=["last_text", "image_patch", "anchor", "image_patch_all", "fixed"],
    )
    p.add_argument("--blend", type=float, default=1.0)
    p.add_argument("--fixed-token-index", type=int, default=None)
    p.add_argument("--image-patch-seed", type=int, default=0)
    p.add_argument("--out-json", default=None, help="Write a small JSON summary here")
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

    if args.placement == "fixed" and args.fixed_token_index is None:
        raise SystemExit("--fixed-token-index is required when --placement=fixed")

    steer_text = _load_steer_text(args)
    mods = _imports()
    embodiment_tag = mods["EmbodimentTag"].resolve(args.embodiment_tag)
    _preflight_processor_access(args.model_path)

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

    spec = SteerSpec(
        placement=args.placement,  # type: ignore[arg-type]
        blend=float(args.blend),
        fixed_token_index=args.fixed_token_index,
        image_patch_seed=int(args.image_patch_seed),
    )

    logger.info("Running baseline policy.get_action …")
    with_policy = policy.model
    if not hasattr(with_policy, "backbone"):
        raise RuntimeError("policy.model has no .backbone; GR00T layout changed?")

    policy.model.eval()

    with torch.inference_mode():
        base_action = policy_get_action(policy, obs)

        logger.info("Running steered forward (hook on backbone) …")
        with attach_backbone_steer(with_policy.backbone, steer_vec, spec):
            steer_action = policy_get_action(policy, obs)

    stats = action_stats(base_action, steer_action)
    logger.info(
        "Global max |Δaction| = %.6f across all returned modality tensors",
        stats["global_max_abs"],
    )
    for k, v in stats["per_modality_key"].items():
        if "error" in v:
            logger.warning("%s: %s", k, v["error"])
        else:
            logger.info(
                "%s  max_abs=%.6f  mean_abs=%.6f  rms=%.6f",
                k,
                v["max_abs"],
                v["mean_abs"],
                v["rms"],
            )

    if args.out_json:
        payload = {
            "placement": args.placement,
            "blend": float(args.blend),
            "traj_id": args.traj_id,
            "step": args.step,
            "action_delta": stats,
        }
        Path(args.out_json).write_text(json.dumps(payload, indent=2))
        logger.info("Wrote %s", args.out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
