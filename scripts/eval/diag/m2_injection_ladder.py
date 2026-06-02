"""Diagnostic 2 — the injection ladder (sim). The bottleneck localizer.

Runs two NEW arms on the same 100 samples (same seeds: seed_i = base + 17*i),
with the policy language HELD at the target intent so the injection effect is
isolated. Compares against the cached baselines already in the n=100 results.

  real_h        : inject the REAL image-patch grid h_grid [128,H] (no AV, no AR)
                  -> the steering CEILING. Is the injection mechanism even potent?
  oracle_caption: inject AR(gold GPT caption) [128,H] (no AV)
                  -> AR given a perfect caption.

Baselines pulled from the cached results JSON (NOT re-simulated):
  matched_semantic (sft_av), matched no_steer, mismatched_source.

Decision (on predicate_rate and mean r_sim):
  - real_h ~ no_steer (both weak)            -> injection ceiling is low: it's
                                                 α/placement, NOT the codec.
  - real_h strong, oracle_caption weak       -> AR is the bottleneck.
  - oracle_caption strong, matched weak      -> AV is the bottleneck.

This is a long, detached job (~1-2h: 200 fresh rollouts at ~58s each / n_workers).
Launch with setsid nohup per CLAUDE.md. Needs the GR00T policy server live.

Usage (detached):
  PYTHONPATH=src .venv/bin/python scripts/eval/diag/m2_injection_ladder.py \
      --sft-dir data/sft/v9_combined_12k --device cuda \
      --policy-host localhost --policy-port 5556 --n-workers 4 \
      --cache-path data/eval/sim_rollout_cache.jsonl \
      --baseline-results data/eval/v9_combined_12k_n100_cf_strided_cached.json \
      --out-json data/eval/diag/m2_ladder_v9_combined_12k.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from nla.training.checkpoint import load_ar_from_sft
from nla.training.sim_reward import SimRewardJob, SimRewardWorker, encode_texts_with_ar
from sample_set import ROOT, load_samples  # script-dir is on sys.path when run as a file


def _baseline_summary(results_path: Path) -> dict:
    d = json.load(open(results_path))
    keys = {
        "matched_semantic": "sft_av",
        "matched_no_steer": "sft_av__no_steer",
        "mismatched_semantic": "sft_av__mismatched_source",
    }
    out = {}
    for name, k in keys.items():
        out[name] = {
            "predicate_rate": d.get(f"{k}_predicate_rate"),
            "mean_r_sim": d.get(f"{k}_mean_r_sim"),
            "n_active": d.get(f"{k}_n_active"),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--policy-host", default="localhost")
    ap.add_argument("--policy-port", type=int, default=5556)
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--sim-batch-size", type=int, default=4)  # >=2 routes to
    # batched_rollout.py, the ONLY path that supports image_patch_strided.
    ap.add_argument("--sim-max-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blend", type=float, default=1.0)
    ap.add_argument("--strided-k", type=int, default=128)
    ap.add_argument("--rollout-python", default=None,
                    help="LIBERO venv python (has gr00t+libero). Defaults to the "
                         "libero_uv venv, matching compare_cf_steer_checkpoints.py.")
    ap.add_argument("--cache-path", default="data/eval/sim_rollout_cache.jsonl")
    ap.add_argument("--baseline-results",
                    default="data/eval/v9_combined_12k_n100_cf_strided_cached.json")
    ap.add_argument("--arms", default="real_h,oracle_caption",
                    help="comma list: real_h, oracle_caption")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    sft_dir = Path(args.sft_dir)
    if not sft_dir.is_absolute():
        sft_dir = ROOT / sft_dir
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    samples = load_samples(limit=args.limit, load_tensors=True)
    if "oracle_caption" in arms:
        samples = [s for s in samples if s.gold_caption]
    print(f"[m2] {len(samples)} samples; arms={arms}", flush=True)

    ar = None
    if "oracle_caption" in arms:
        ar = load_ar_from_sft(sft_dir / "ar", device=args.device, freeze=True)

    jobs: list[SimRewardJob] = []
    meta: list[dict] = []
    for i, s in enumerate(samples):
        seed = args.seed + i * 17  # MATCH compare_cf_steer_checkpoints.py seeding
        for arm in arms:
            if arm == "real_h":
                steer = np.ascontiguousarray(s.h_grid, dtype=np.float32)  # [128,H]
                text = s.target_intent
            elif arm == "oracle_caption":
                v = encode_texts_with_ar(ar, [s.gold_caption], device=args.device)
                steer = v.reshape(-1, v.shape[-1]).numpy().astype(np.float32)  # [128,H]
                text = s.gold_caption
            else:
                raise ValueError(f"unknown arm {arm!r}")
            jobs.append(SimRewardJob(
                env_name=s.target_env_name,
                target_task=s.target_task,
                source_id=s.sid,
                text=text,
                seed=seed,
                steer_h=steer,
                sim_max_steps=args.sim_max_steps,
                placement="image_patch_strided",
                blend=float(args.blend),
                policy_language_override=s.target_intent,  # language HELD at target
                steer_disabled=False,
                strided_k=int(args.strided_k),
            ))
            meta.append({"sid": s.sid, "arm": arm, "target_task": s.target_task})

    libero_py = args.rollout_python or str(
        ROOT / "third_party/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"
    )
    worker = SimRewardWorker(
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        n_workers=args.n_workers,
        sim_batch_size=args.sim_batch_size,
        rollout_python=libero_py,
        cache_path=(ROOT / args.cache_path) if not Path(args.cache_path).is_absolute()
        else args.cache_path,
    )
    print(f"[m2] worker ready; cache has {len(worker._cache)} entries; "
          f"running {len(jobs)} rollouts", flush=True)

    t0 = time.time()
    res = worker.compute(jobs)
    print(f"[m2] {len(jobs)} rollouts done in {time.time()-t0:.0f}s", flush=True)

    # aggregate per arm
    per_arm: dict[str, list] = {a: [] for a in arms}
    per_sample = []
    for r, m in zip(res, meta):
        per_arm[m["arm"]].append(r)
        per_sample.append({**m, "r_sim": r.r_sim, "predicate": r.predicate,
                           "success_any": r.success_any, "error": r.error})

    summary = {"sft_dir": str(sft_dir), "n_samples": len(samples),
               "seed_base": args.seed, "arms": {}}
    for a in arms:
        rows = [r for r in per_arm[a] if r.error is None]
        n = max(1, len(rows))
        summary["arms"][a] = {
            "predicate_rate": sum(1 for r in rows if r.predicate > 0) / n,
            "mean_r_sim": sum(r.r_sim for r in rows) / n,
            "n_active": len(rows),
            "n_error": len(per_arm[a]) - len(rows),
        }
    summary["baselines_cached"] = _baseline_summary(
        (ROOT / args.baseline_results) if not Path(args.baseline_results).is_absolute()
        else Path(args.baseline_results))
    summary["per_sample"] = per_sample

    out = Path(args.out_json)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print("\n=== Diagnostic 2: injection ladder (pred_rate / mean r_sim) ===")
    b = summary["baselines_cached"]
    print(f"  [ceiling] real_h         : {summary['arms'].get('real_h', {})}")
    print(f"  [AR|gold] oracle_caption : {summary['arms'].get('oracle_caption', {})}")
    print(f"  matched_semantic (base)  : {b['matched_semantic']}")
    print(f"  matched_no_steer (base)  : {b['matched_no_steer']}")
    print(f"  mismatched       (base)  : {b['mismatched_semantic']}")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
