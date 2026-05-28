#!/usr/bin/env python
"""Action-effect probe — gate before launching GRPO v7.

The v7 runbook prescribes this as the first eval after SFT v7 completes:

    > For each held-out CF pair, compute next-action delta between
    > policy(h) and policy(h_steered) with AR(AV(h)). Median delta ≥
    > threshold ⇒ codec is causally connected to the policy ⇒ GRPO is
    > worth running.

We don't have a probability distribution to compute KL on (GR00T's action
head emits continuous joint deltas), so we use the L2-of-action-delta
analog: ``rms(steered_action − baseline_action)`` averaged across action
keys, taken median over the held-out CF samples.

Operational gate:
  median rms ≥ 0.05 (action units; LIBERO joints normalize to roughly ±1)
                   → codec produces measurable policy-effect → run GRPO.
  median rms < 0.05 → codec output is still inert → diagnose SFT loss
                   balance before spending GRPO compute.

This script avoids the steer server (since it loads GR00T inline) and the
LIBERO simulator (no rollouts; one forward per pair). Total runtime for
n=32 pairs should be < 10 minutes on a single H100.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/eval/action_effect_probe.py \\
        --sft-dir          data/sft/v7_libero_4suite \\
        --activations-root data/activations/libero_4suite_v4_combined \\
        --pairs-path       data/grpo/libero_goal_counterfactual_pairs_cfonly.jsonl \\
        --model-path       checkpoints/GR00T-N1.7-LIBERO/libero_goal \\
        --embodiment-tag   LIBERO_PANDA \\
        --dataset-path     third_party/Isaac-GR00T/examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot \\
        --n-samples        32 \\
        --out-json         data/sft/v7_libero_4suite/action_effect_probe.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sft-dir", required=True, help="SFT dir with av/ + ar/")
    p.add_argument("--activations-root", required=True)
    p.add_argument("--pairs-path", required=True,
                   help="CF pairs JSONL; rows must carry source_example_id, "
                        "position_index, position_type, target_intent.")
    p.add_argument("--model-path", required=True, help="GR00T checkpoint dir")
    p.add_argument("--embodiment-tag", required=True)
    p.add_argument("--dataset-path", required=True,
                   help="LeRobot root for the suite the held-out pairs come "
                        "from (used to reconstruct policy observations).")
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--placement", default="image_patch",
                   choices=["image_patch", "image_patch_all", "image_patch_spatial",
                            "image_patch_strided", "last_text", "anchor"])
    p.add_argument("--blend", type=float, default=1.0)
    p.add_argument("--video-backend", default="torchcodec",
                   choices=["decord", "torchvision_av", "torchcodec"])
    p.add_argument("--rms-threshold", type=float, default=0.05,
                   help="Pass threshold on the median rms(action delta). "
                        "Median < this ⇒ steer is inert; don't launch GRPO.")
    p.add_argument("--random-control", action="store_true",
                   help="Replace AR's output with a random gaussian vector of "
                        "the same shape and L2 norm. Tests whether the codec's "
                        "content matters, vs. any vector of similar magnitude.")
    p.add_argument("--out-json", required=True)
    return p


def _load_pairs(path: Path, n: int, seed: int) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("target_intent"):
                continue
            if r.get("position_type") != "image_patch":
                continue
            rows.append(r)
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    return rows[:n]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Action-effect probe — sft={args.sft_dir}  n={args.n_samples}", flush=True)
    print(f"  threshold: median rms ≥ {args.rms_threshold} ⇒ PASS", flush=True)

    pairs = _load_pairs(Path(args.pairs_path), args.n_samples, args.seed)
    if not pairs:
        print("FATAL: no usable pairs", file=sys.stderr)
        return 2
    print(f"  loaded {len(pairs)} CF pairs", flush=True)

    # Lazy imports — heavy modules
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    from nla.extraction._compat import apply_all as apply_groot_compat
    from nla.extraction.storage import ActivationShardReader
    from nla.steering import (
        SteerSpec,
        action_stats,
        ar_text_to_backbone_vec,
        attach_backbone_steer,
        policy_get_action,
    )
    from nla.steering.groot_obs import build_observation_for_step
    from nla.training.checkpoint import load_ar_from_sft, load_av_from_sft

    apply_groot_compat()

    embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    print(f"  loading GR00T policy at {args.model_path} ...", flush=True)
    policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=args.model_path,
        device=args.device,
    )
    policy.model.eval()

    modality_configs = deepcopy(policy.modality_configs)
    modality_configs.pop("action", None)
    language_keys = list(policy.modality_configs["language"].modality_keys)
    loader = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=policy.modality_configs,
        video_backend=args.video_backend,
    )

    print(f"  loading AV+AR from {args.sft_dir} ...", flush=True)
    sft_dir = Path(args.sft_dir)
    av = load_av_from_sft(sft_dir / "av", device=args.device, freeze=True)
    ar = load_ar_from_sft(sft_dir / "ar", device=args.device, freeze=True)
    reader = ActivationShardReader(args.activations_root)

    backbone = policy.model.backbone

    per_sample: list[dict] = []
    rms_values: list[float] = []
    t0 = time.time()

    for i, row in enumerate(pairs):
        sid = row["source_example_id"]
        item = reader.get(sid)
        pos = int(row["position_index"])
        ptype = row["position_type"]
        h = item["features"][pos].contiguous().to(torch.float32).to(args.device)

        # Build the observation matching this activation's episode/step.
        episode = row.get("episode_index")
        step = row.get("step_index")
        if episode is None or step is None:
            # Parse from source_example_id like "goal__traj000396_step000040"
            parts = sid.split("_")
            try:
                episode = int([p for p in parts if p.startswith("traj")][0][4:])
                step = int([p for p in parts if p.startswith("step")][0][4:])
            except (IndexError, ValueError):
                print(f"  [{i}] skip: cannot parse episode/step from {sid}", flush=True)
                continue
        try:
            traj = loader[int(episode)]
            obs = build_observation_for_step(
                traj, int(step), modality_configs, embodiment_tag,
                language_keys, extract_step_data,
            )
        except Exception as e:
            print(f"  [{i}] skip: obs build failed for {sid}: {e}", flush=True)
            continue

        # 1) baseline action (no steer)
        with torch.inference_mode():
            baseline = policy_get_action(policy, obs)

        # 2) AR-decoded steer vector from the target intent
        with torch.no_grad():
            av_out = av.generate(
                h.unsqueeze(0),
                [ptype],
                max_new_tokens=160,
                temperature=0.0,
                do_sample=False,
                target_intent_texts=[row["target_intent"]],
            )
            text = av_out["text"][0]
            steer_vec = ar_text_to_backbone_vec(ar, text).to(args.device)
            per_position_placements = ("image_patch_spatial", "image_patch_strided")
            if steer_vec.dim() == 2 and args.placement not in per_position_placements:
                # Spatial AR head emits (K, H); single-position placements
                # take (H,), so collapse via mean-pool. Per-position
                # placements keep the (K, H) shape.
                steer_vec = steer_vec.mean(dim=0)
            if args.random_control:
                # Replace AR's actual output with a random gaussian vector
                # rescaled to AR's L2 norm. Tests whether image_patch_all PASS
                # depends on AR's content or just on broadcasting any vector.
                orig_norm = steer_vec.norm()
                rand = torch.randn_like(steer_vec)
                if steer_vec.dim() == 2:
                    # Match per-row norms so the random control's magnitude
                    # profile across K matches AR's.
                    rand = rand / rand.norm(dim=1, keepdim=True) * steer_vec.norm(dim=1, keepdim=True)
                else:
                    rand = rand / rand.norm() * orig_norm
                steer_vec = rand
            strided_k = int(steer_vec.shape[0]) if (
                args.placement == "image_patch_strided" and steer_vec.dim() == 2
            ) else 0

        # 3) steered action
        spec = SteerSpec(
            placement=args.placement, blend=float(args.blend),
            image_patch_seed=int(args.seed) + i,
            strided_k=strided_k,
        )
        with torch.inference_mode():
            with attach_backbone_steer(backbone, steer_vec, spec):
                steered = policy_get_action(policy, obs)

        # 4) action delta
        delta = action_stats(baseline, steered)
        rmses = [
            v["rms"] for v in delta.get("per_modality_key", {}).values()
            if isinstance(v, dict) and "rms" in v
        ]
        sample_rms = float(np.mean(rmses)) if rmses else 0.0
        rms_values.append(sample_rms)
        per_sample.append({
            "source_example_id": sid,
            "target_intent": row["target_intent"][:80],
            "rms_mean_over_keys": sample_rms,
            "global_max_abs": float(delta.get("global_max_abs", 0.0)),
            "av_text_preview": text[:160].replace("\n", " "),
        })

        if (i + 1) % 4 == 0 or i + 1 == len(pairs):
            elapsed = time.time() - t0
            median_so_far = statistics.median(rms_values) if rms_values else 0.0
            print(
                f"  [{i+1}/{len(pairs)}] last_rms={sample_rms:.4f}  "
                f"running_median={median_so_far:.4f}  elapsed={elapsed/60:.1f}min",
                flush=True,
            )

    if not rms_values:
        print("FATAL: zero usable samples", file=sys.stderr)
        return 3

    median_rms = float(statistics.median(rms_values))
    mean_rms = float(np.mean(rms_values))
    p25_rms = float(np.percentile(rms_values, 25))
    p75_rms = float(np.percentile(rms_values, 75))
    median_gmaxabs = float(statistics.median(
        s["global_max_abs"] for s in per_sample
    ))

    verdict = "PASS" if median_rms >= args.rms_threshold else "FAIL"
    summary = {
        "verdict": verdict,
        "threshold": float(args.rms_threshold),
        "n_samples": len(rms_values),
        "median_rms": median_rms,
        "mean_rms": mean_rms,
        "p25_rms": p25_rms,
        "p75_rms": p75_rms,
        "median_global_max_abs": median_gmaxabs,
        "placement": args.placement,
        "blend": args.blend,
        "sft_dir": args.sft_dir,
        "per_sample": per_sample,
    }
    out_path.write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 60)
    print(f"Action-effect probe — {verdict}")
    print("=" * 60)
    print(f"  median rms(action delta):    {median_rms:.4f}  (threshold {args.rms_threshold:.3f})")
    print(f"  mean / p25 / p75:            {mean_rms:.4f} / {p25_rms:.4f} / {p75_rms:.4f}")
    print(f"  median global_max_abs:       {median_gmaxabs:.4f}")
    print(f"  n samples:                   {len(rms_values)}")
    print(f"  -> {out_path}")
    if verdict == "PASS":
        print(f"  Codec is causally connected to the policy. Proceed to GRPO v7.")
    else:
        print(f"  Codec output is still inert. Diagnose SFT loss balance before GRPO.")
    return 0 if verdict == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
