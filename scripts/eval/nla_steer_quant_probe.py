#!/usr/bin/env python
"""**Quantitative** steer probe — one timestep, **two** steer prompts, printed math.

Use this when you want **numbers**, not opinions about MP4 pixels.

For a **single** dataset frame (``--traj-id`` / ``--step``) we:

1. Build AR vectors **ĥ_a**, **ĥ_b** from **two different bullet prompts** (different language → different backbone injections).
2. Run ``get_action`` **three ways** on the **same** observation:
   **baseline**, **steer(a)**, **steer(b)** (each with optional torch seed).
3. Print **exact previews** (first ``--show-dims`` flat scalars per modality key) and
   ``action_stats`` deltas:

   - baseline → steer(a)
   - baseline → steer(b)
   - steer(a) → steer(b)  (**different prompts truly pull the policy differently**)

Also prints **‖ĥ_a‖₂**, **‖ĥ_b‖₂**, **cos(ĥ_a, ĥ_b)**.

This does **not** use Cosmos as a video world model — GR00T loads Cosmos Reason as the **VL backbone**, not as “predict next RGB frames” here.

Example (LIBERO)::

    PYTHONPATH=src .venv/bin/python scripts/eval/nla_steer_quant_probe.py \\
        --model-path     checkpoints/GR00T-N1.7-LIBERO/libero_goal \\
        --dataset-path   third_party/Isaac-GR00T/examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot \\
        --embodiment-tag LIBERO_PANDA \\
        --ar-dir         data/sft/libero_goal_pilot_v3/ar \\
        --traj-id 0 --step 50 \\
        --placement      anchor --blend 1.0 \\
        --seed 0 \\
        --steer-text-a-file prompts/a_push_left.txt \\
        --steer-text-b-file prompts/b_push_right.txt \\
        --out-json steer_quant.json
"""

from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch


logger = logging.getLogger("nla.steer_quant")


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


def _load_text(arg: str | None, path: str | None, *, role: str) -> str:
    if arg and path:
        raise SystemExit(f"Use only one of text / file for {role}")
    if path:
        return Path(path).read_text().strip()
    if arg:
        return arg.strip()
    raise SystemExit(f"Provide --steer-text-{role} or --steer-text-{role}-file")


def _vec_report(h_a: np.ndarray, h_b: np.ndarray) -> dict[str, float]:
    ha = h_a.astype(np.float64).ravel()
    hb = h_b.astype(np.float64).ravel()
    na = float(np.linalg.norm(ha))
    nb = float(np.linalg.norm(hb))
    cos = float(np.dot(ha, hb) / max(na * nb, 1e-12))
    return {"l2_a": na, "l2_b": nb, "cos_ab": cos, "l2_diff": float(np.linalg.norm(ha - hb))}


def _preview_vectors(
    baseline: dict[str, Any],
    sa: dict[str, Any],
    sb: dict[str, Any],
    *,
    show_dims: int,
    to_numpy,
) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {}
    keys = sorted(set(baseline.keys()) | set(sa.keys()) | set(sb.keys()))
    for k in keys:
        b = to_numpy(baseline.get(k)).ravel()
        a = to_numpy(sa.get(k)).ravel()
        b2 = to_numpy(sb.get(k)).ravel()
        if b.shape != a.shape or b.shape != b2.shape:
            continue
        n = min(show_dims, int(b.size))
        out[k] = {
            "baseline": b[:n].astype(float).tolist(),
            "steer_a": a[:n].astype(float).tolist(),
            "steer_b": b2[:n].astype(float).tolist(),
            "delta_a_minus_base": (a[:n] - b[:n]).astype(float).tolist(),
            "delta_b_minus_base": (b2[:n] - b[:n]).astype(float).tolist(),
            "delta_b_minus_a": (b2[:n] - a[:n]).astype(float).tolist(),
        }
    return out


def _print_previews(previews: dict[str, dict[str, list[float]]], *, max_keys: int) -> None:
    print("\n=== Per-modality previews (first K floats, flat) ===\n")
    for i, (k, blk) in enumerate(previews.items()):
        if i >= max_keys:
            print(f"... ({len(previews) - max_keys} more keys omitted; raise --print-keys)")
            break
        print(f"[{k}]")
        for label in (
            "baseline",
            "steer_a",
            "steer_b",
            "delta_a_minus_base",
            "delta_b_minus_base",
            "delta_b_minus_a",
        ):
            arr = blk.get(label)
            if arr is None:
                continue
            print(f"  {label:22s} {np.array2string(np.asarray(arr), precision=6, floatmode='fixed')}")
        print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model-path", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--embodiment-tag", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ar-dir", required=True)
    p.add_argument("--traj-id", type=int, required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--video-backend", default="torchcodec",
                   choices=["decord", "torchvision_av", "torchcodec"])
    p.add_argument("--steer-text-a", default=None)
    p.add_argument("--steer-text-a-file", default=None)
    p.add_argument("--steer-text-b", default=None)
    p.add_argument("--steer-text-b-file", default=None)
    p.add_argument(
        "--ar-position-type",
        default=None,
        choices=["last_text", "image_patch", "anchor", "fallback"],
        help="Position type for AR context_v5 prompt (required when the "
             "checkpoint was trained with --ar-prompt-version=context_v5).",
    )
    p.add_argument(
        "--step-index",
        type=int,
        default=None,
        help="Timestep for AR context_v5 prompt (defaults to dataset step "
             "when --ar-position-type is set).",
    )
    p.add_argument(
        "--instruction",
        default=None,
        help="Task instruction string for AR context_v5 prompt.",
    )
    p.add_argument(
        "--placement",
        default="anchor",
        choices=["last_text", "image_patch", "anchor", "image_patch_all", "fixed"],
    )
    p.add_argument("--blend", type=float, default=1.0)
    p.add_argument("--fixed-token-index", type=int, default=None)
    p.add_argument("--image-patch-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--show-dims", type=int, default=12)
    p.add_argument("--print-keys", type=int, default=8, help="Max modality keys to print to stdout")
    p.add_argument("--out-json", default=None)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    from nla.steering import (
        SteerSpec,
        action_stats,
        ar_text_to_backbone_vec,
        attach_backbone_steer,
        policy_get_action,
        to_numpy,
    )
    from nla.steering.groot_obs import build_observation_for_step
    from nla.training.checkpoint import load_ar_from_sft

    if args.placement == "fixed" and args.fixed_token_index is None:
        raise SystemExit("--fixed-token-index required when --placement=fixed")

    ta = _load_text(args.steer_text_a, args.steer_text_a_file, role="a")
    tb = _load_text(args.steer_text_b, args.steer_text_b_file, role="b")

    mods = _imports()
    embodiment_tag = mods["EmbodimentTag"].resolve(args.embodiment_tag)

    policy = mods["Gr00tPolicy"](
        embodiment_tag=embodiment_tag,
        model_path=args.model_path,
        device=args.device,
    )
    policy.model.eval()

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

    ar = load_ar_from_sft(Path(args.ar_dir), device=args.device, freeze=True)
    ar_ctx: dict = {}
    if args.ar_position_type is not None:
        ar_ctx["position_type"] = args.ar_position_type
        ar_ctx["step_index"] = args.step_index if args.step_index is not None else args.step
        if args.instruction is not None:
            ar_ctx["instruction"] = args.instruction
    v_a = ar_text_to_backbone_vec(ar, ta, **ar_ctx).detach().cpu().numpy()
    v_b = ar_text_to_backbone_vec(ar, tb, **ar_ctx).detach().cpu().numpy()
    vec_rep = _vec_report(v_a, v_b)

    spec = SteerSpec(
        placement=args.placement,  # type: ignore[arg-type]
        blend=float(args.blend),
        fixed_token_index=args.fixed_token_index,
        image_patch_seed=int(args.image_patch_seed),
    )

    backbone = policy.model.backbone
    steer_a_t = torch.from_numpy(v_a).to(args.device)
    steer_b_t = torch.from_numpy(v_b).to(args.device)

    def run(hook_vec: torch.Tensor | None) -> dict[str, Any]:
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
        if hook_vec is None:
            return policy_get_action(policy, obs)
        with attach_backbone_steer(backbone, hook_vec, spec):
            return policy_get_action(policy, obs)

    with torch.inference_mode():
        baseline = run(None)
        steer_a = run(steer_a_t)
        steer_b = run(steer_b_t)

    s0a = action_stats(baseline, steer_a)
    s0b = action_stats(baseline, steer_b)
    sab = action_stats(steer_a, steer_b)

    previews = _preview_vectors(baseline, steer_a, steer_b, show_dims=int(args.show_dims), to_numpy=to_numpy)

    print("\n=== AR(backbone) vectors (prompt separation) ===")
    print(json.dumps(vec_rep, indent=2))
    print("\n=== Δaction stats: baseline → steer(A) ===")
    print(json.dumps(s0a, indent=2))
    print("\n=== Δaction stats: baseline → steer(B) ===")
    print(json.dumps(s0b, indent=2))
    print("\n=== Δaction stats: steer(A) → steer(B)  (contrasting prompts) ===")
    print(json.dumps(sab, indent=2))

    _print_previews(previews, max_keys=int(args.print_keys))

    print(
        "\nInterpret: large ``global_max_abs`` under baseline→steer shows backbone patch moves ``get_action``. "
        "Non‑tiny steer(A)→steer(B) shows **different language maps (via AR) to measurably different policy outputs** "
        "on this frozen observation.\n"
    )

    if args.out_json:
        payload = {
            "traj_id": args.traj_id,
            "step": args.step,
            "placement": args.placement,
            "blend": float(args.blend),
            "seed": int(args.seed),
            "ar_vectors": vec_rep,
            "delta_baseline_to_a": s0a,
            "delta_baseline_to_b": s0b,
            "delta_a_to_b": sab,
            "previews": previews,
            "steer_text_a": ta[:500],
            "steer_text_b": tb[:500],
        }
        Path(args.out_json).write_text(json.dumps(payload, indent=2))
        logger.info("Wrote %s", args.out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
