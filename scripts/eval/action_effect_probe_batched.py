#!/usr/bin/env python
"""Batched action-effect probe.

Runs multiple (placement, random_control) probe variants against a single
SFT checkpoint while paying the model-loading cost (~20 min AV embedding
init + policy load) ONCE.

Use this instead of running scripts/eval/action_effect_probe.py 4 times
in a row when you want a side-by-side comparison of:
  - codec vs random control (does codec content matter or just magnitude?)
  - single-position vs broadcast vs spatial injection

Usage::

    PYTHONPATH=src .venv/bin/python scripts/eval/action_effect_probe_batched.py \\
        --sft-dir          data/sft/v8_libero_4suite \\
        --activations-root data/activations/libero_4suite_v4_combined \\
        --pairs-path       data/grpo/libero_goal_counterfactual_pairs_cfonly.jsonl \\
        --model-path       checkpoints/GR00T-N1.7-LIBERO/libero_goal \\
        --embodiment-tag   LIBERO_PANDA \\
        --dataset-path     third_party/Isaac-GR00T/examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot \\
        --n-samples        32 \\
        --variants         image_patch_all,image_patch_all+random,image_patch_spatial,image_patch_spatial+random \\
        --out-dir          data/sft/v8_libero_4suite/probes_batched
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


VARIANT_PLACEMENTS = {
    "image_patch",
    "image_patch_all",
    "image_patch_spatial",
    "image_patch_strided",
    "last_text",
    "anchor",
}


@dataclass
class Variant:
    """One probe configuration: a placement plus optional random control."""
    name: str        # e.g. "image_patch_all" or "image_patch_all+random"
    placement: str   # the SteerSpec placement
    random_control: bool  # if True, replace AR's output with a magnitude-matched gaussian

    @classmethod
    def parse(cls, spec: str) -> "Variant":
        """Parse 'image_patch_all' or 'image_patch_all+random' format."""
        random_control = spec.endswith("+random")
        placement = spec[:-len("+random")] if random_control else spec
        if placement not in VARIANT_PLACEMENTS:
            raise SystemExit(
                f"Unknown placement {placement!r} in variant spec {spec!r}. "
                f"Allowed: {sorted(VARIANT_PLACEMENTS)}"
            )
        return cls(name=spec, placement=placement, random_control=random_control)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sft-dir", required=True)
    p.add_argument("--activations-root", required=True)
    p.add_argument("--pairs-path", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--embodiment-tag", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--blend", type=float, default=1.0)
    p.add_argument("--video-backend", default="torchcodec",
                   choices=["decord", "torchvision_av", "torchcodec"])
    p.add_argument("--rms-threshold", type=float, default=0.05)
    p.add_argument(
        "--variants",
        default="image_patch_all,image_patch_all+random,image_patch_spatial,image_patch_spatial+random",
        help="Comma-separated list of variant specs. Each is a placement "
             "name optionally followed by '+random' for the random-control "
             "version. e.g. 'image_patch_all,image_patch_all+random'.",
    )
    p.add_argument(
        "--out-dir", required=True,
        help="Output directory. Each variant writes <variant_name>.json here.",
    )
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
            rows.append(r)
    rng = np.random.default_rng(seed)
    if len(rows) > n:
        idx = rng.choice(len(rows), size=n, replace=False)
        rows = [rows[int(i)] for i in idx]
    return rows


def _run_one_sample(
    av, ar, policy, backbone, loader, modality_configs, embodiment_tag,
    language_keys, extract_step_data, build_observation_for_step, reader,
    SteerSpec, ar_text_to_backbone_vec, attach_backbone_steer,
    action_stats, policy_get_action,
    pair, variant, i, args,
) -> tuple[float | None, dict | None]:
    """Run one sample for one variant. Returns (rms_value, per_sample_dict) or (None, None) on skip."""
    sid = pair["source_example_id"]
    item = reader.get(sid)
    pos = int(pair["position_index"])
    ptype = pair["position_type"]
    h = item["features"][pos].contiguous().to(torch.float32).to(args.device)

    episode = pair.get("episode_index")
    step = pair.get("step_index")
    if episode is None or step is None:
        parts = sid.split("_")
        try:
            episode = int([p for p in parts if p.startswith("traj")][0][4:])
            step = int([p for p in parts if p.startswith("step")][0][4:])
        except (IndexError, ValueError):
            return None, None
    try:
        traj = loader[int(episode)]
        obs = build_observation_for_step(
            traj, int(step), modality_configs, embodiment_tag,
            language_keys, extract_step_data,
        )
    except Exception:
        return None, None

    # 1) baseline action
    with torch.inference_mode():
        baseline = policy_get_action(policy, obs)

    # 2) AR-decoded steer vector
    with torch.no_grad():
        av_out = av.generate(
            h.unsqueeze(0),
            [ptype],
            max_new_tokens=160,
            temperature=0.0,
            do_sample=False,
            target_intent_texts=[pair["target_intent"]],
        )
        text = av_out["text"][0]
        steer_vec = ar_text_to_backbone_vec(ar, text).to(args.device)
        per_position_placements = ("image_patch_spatial", "image_patch_strided")
        if steer_vec.dim() == 2 and variant.placement not in per_position_placements:
            steer_vec = steer_vec.mean(dim=0)
        if variant.random_control:
            orig_norm = steer_vec.norm()
            rand = torch.randn_like(steer_vec)
            if steer_vec.dim() == 2:
                rand = rand / rand.norm(dim=1, keepdim=True) * steer_vec.norm(dim=1, keepdim=True)
            else:
                rand = rand / rand.norm() * orig_norm
            steer_vec = rand
        strided_k = int(steer_vec.shape[0]) if (
            variant.placement == "image_patch_strided" and steer_vec.dim() == 2
        ) else 0

    # 3) steered action
    spec = SteerSpec(
        placement=variant.placement, blend=float(args.blend),
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
    per_sample = {
        "source_example_id": sid,
        "target_intent": pair["target_intent"][:80],
        "rms_mean_over_keys": sample_rms,
        "global_max_abs": float(delta.get("global_max_abs", 0.0)),
        "av_text_preview": text[:160].replace("\n", " "),
    }
    return sample_rms, per_sample


def _summarize_and_save(
    variant: Variant, rms_values: list[float], per_sample: list[dict],
    out_dir: Path, sft_dir: str, blend: float, threshold: float,
) -> dict:
    if not rms_values:
        summary = {"verdict": "ERROR", "n_samples": 0, "variant": variant.name}
        (out_dir / f"{variant.name.replace('+','_')}.json").write_text(json.dumps(summary, indent=2))
        return summary

    median_rms = float(statistics.median(rms_values))
    mean_rms = float(np.mean(rms_values))
    p25 = float(np.percentile(rms_values, 25))
    p75 = float(np.percentile(rms_values, 75))
    median_gmaxabs = float(statistics.median(s["global_max_abs"] for s in per_sample))
    verdict = "PASS" if median_rms >= threshold else "FAIL"

    summary = {
        "verdict": verdict,
        "variant": variant.name,
        "placement": variant.placement,
        "random_control": variant.random_control,
        "threshold": threshold,
        "n_samples": len(rms_values),
        "median_rms": median_rms,
        "mean_rms": mean_rms,
        "p25_rms": p25,
        "p75_rms": p75,
        "median_global_max_abs": median_gmaxabs,
        "blend": blend,
        "sft_dir": sft_dir,
        "per_sample": per_sample,
    }
    out_path = out_dir / f"{variant.name.replace('+','_')}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = [Variant.parse(s.strip()) for s in args.variants.split(",") if s.strip()]
    if not variants:
        raise SystemExit("--variants must list at least one probe variant")

    print(f"Batched probe — sft={args.sft_dir}  n={args.n_samples}", flush=True)
    print(f"  variants: {[v.name for v in variants]}", flush=True)
    print(f"  threshold: median rms ≥ {args.rms_threshold} ⇒ PASS", flush=True)

    pairs = _load_pairs(Path(args.pairs_path), args.n_samples, args.seed)
    if not pairs:
        print("FATAL: no usable pairs", file=sys.stderr)
        return 2
    print(f"  loaded {len(pairs)} CF pairs", flush=True)

    # Lazy heavy imports
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

    # ---- LOAD MODELS ONCE ----
    embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    print(f"  loading GR00T policy at {args.model_path} ...", flush=True)
    t_load = time.time()
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
    print(f"  policy loaded in {time.time()-t_load:.1f}s", flush=True)

    t_load = time.time()
    print(f"  loading AV+AR from {args.sft_dir} (this triggers the slow MVN init) ...", flush=True)
    sft_dir = Path(args.sft_dir)
    av = load_av_from_sft(sft_dir / "av", device=args.device, freeze=True)
    ar = load_ar_from_sft(sft_dir / "ar", device=args.device, freeze=True)
    print(f"  AV+AR loaded in {time.time()-t_load:.1f}s", flush=True)
    reader = ActivationShardReader(args.activations_root)
    backbone = policy.model.backbone

    # ---- RUN EACH VARIANT ----
    all_summaries: dict[str, dict] = {}
    for v_idx, variant in enumerate(variants):
        print(
            f"\n=== variant {v_idx+1}/{len(variants)}: {variant.name} "
            f"(placement={variant.placement}, random_control={variant.random_control}) ===",
            flush=True,
        )
        rms_values: list[float] = []
        per_sample: list[dict] = []
        t0 = time.time()
        for i, pair in enumerate(pairs):
            try:
                rms, sample = _run_one_sample(
                    av, ar, policy, backbone, loader, modality_configs,
                    embodiment_tag, language_keys, extract_step_data,
                    build_observation_for_step, reader,
                    SteerSpec, ar_text_to_backbone_vec,
                    attach_backbone_steer, action_stats, policy_get_action,
                    pair, variant, i, args,
                )
            except Exception as e:
                print(f"  [{i}] ERROR on {pair['source_example_id']}: {e}", flush=True)
                continue
            if rms is None:
                continue
            rms_values.append(rms)
            per_sample.append(sample)
            if (i + 1) % 4 == 0 or i + 1 == len(pairs):
                elapsed = time.time() - t0
                median_so_far = statistics.median(rms_values) if rms_values else 0.0
                print(
                    f"  [{i+1}/{len(pairs)}] last_rms={rms:.4f}  "
                    f"running_median={median_so_far:.4f}  elapsed={elapsed/60:.1f}min",
                    flush=True,
                )
        summary = _summarize_and_save(
            variant, rms_values, per_sample, out_dir,
            sft_dir=str(sft_dir), blend=args.blend, threshold=args.rms_threshold,
        )
        all_summaries[variant.name] = summary
        print(
            f"  → {summary['verdict']}  median_rms={summary['median_rms']:.4f}  "
            f"(n={summary['n_samples']})",
            flush=True,
        )

    # ---- FINAL SUMMARY TABLE ----
    print("\n" + "=" * 70, flush=True)
    print("Batched probe complete. Summary:", flush=True)
    print("=" * 70, flush=True)
    print(f"{'variant':<40} {'median':>8} {'mean':>8} {'verdict':>8}", flush=True)
    print("-" * 70, flush=True)
    for name, s in all_summaries.items():
        if s["n_samples"] == 0:
            print(f"{name:<40}  (no samples)", flush=True)
            continue
        print(
            f"{name:<40} {s['median_rms']:>8.4f} {s['mean_rms']:>8.4f} {s['verdict']:>8}",
            flush=True,
        )
    print(f"\nJSON outputs in: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
