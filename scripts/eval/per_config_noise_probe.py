"""Measure per-config rollout noise on the matched/semantic arm.

For each of 3 source_ids, run N=10 rollouts varying only the LIBERO seed.
Same (sample, intent, AV decode, AR encode, placement) — only sim-side
randomness differs. Reports per-config r_sim std and compares it to the
between-config std from the n=100 CF eval (0.099).

Usage:
  PYTHONPATH=src python scripts/eval/per_config_noise_probe.py \
    --sft-dir data/sft/v9_combined_12k \
    --pairs-path data/grpo/libero_goal_counterfactual_pairs_cfonly.jsonl \
    --activations-root data/activations/libero_4suite_v4_combined \
    --source-ids goal__traj000162_step000038,goal__traj000398_step000006,goal__traj000310_step000036 \
    --n-reseeds 10 \
    --policy-host localhost --policy-port 5556 \
    --out-json data/eval/v9_combined_12k_noise_probe.json
"""
from __future__ import annotations
import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-dir", required=True)
    ap.add_argument("--pairs-path", required=True)
    ap.add_argument("--activations-root", required=True)
    ap.add_argument("--source-ids", required=True,
                    help="Comma-separated source_example_ids to probe")
    ap.add_argument("--n-reseeds", type=int, default=10)
    ap.add_argument("--policy-host", default="localhost")
    ap.add_argument("--policy-port", type=int, default=5556)
    ap.add_argument("--sim-cache-path", default="data/eval/sim_rollout_cache.jsonl")
    ap.add_argument("--sim-max-steps", type=int, default=100)
    ap.add_argument("--sim-placement", default="image_patch_strided")
    ap.add_argument("--strided-k", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--sim-batch-size", type=int, default=4)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    from nla.extraction.storage import ActivationShardReader
    from nla.training.sim_reward import SimRewardWorker, assemble_jobs, encode_texts_with_ar
    from nla.training.checkpoint import load_ar_from_sft, load_av_from_sft

    sft_dir = Path(args.sft_dir)
    target_ids = [s.strip() for s in args.source_ids.split(",") if s.strip()]
    print(f"[probe] target source_ids: {target_ids}", flush=True)

    # Load pairs and pick rows
    rows: dict[str, dict] = {}
    with open(args.pairs_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            sid = obj.get("source_example_id")
            if sid in target_ids and sid not in rows:
                rows[sid] = obj
            if len(rows) == len(target_ids):
                break
    missing = [s for s in target_ids if s not in rows]
    if missing:
        print(f"FATAL: source_ids not found in pairs: {missing}", file=sys.stderr)
        return 2

    t0 = time.time()
    print(f"[probe] loading activations reader from {args.activations_root}", flush=True)
    reader = ActivationShardReader(args.activations_root)
    print(f"[probe] reader loaded ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    print(f"[probe] loading AR from {sft_dir / 'ar'}", flush=True)
    ar = load_ar_from_sft(sft_dir / "ar", device=args.device, freeze=True)
    print(f"[probe] AR loaded ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    print(f"[probe] loading AV from {sft_dir / 'av'}", flush=True)
    av = load_av_from_sft(sft_dir / "av", device=args.device, freeze=True)
    print(f"[probe] AV loaded ({time.time()-t0:.1f}s)", flush=True)

    # Build steer vec per sample (matched intent only)
    per_sample_text: dict[str, str] = {}
    per_sample_steer: dict[str, torch.Tensor] = {}
    per_sample_meta: dict[str, dict] = {}
    for sid in target_ids:
        row = rows[sid]
        task = row["target_task"]
        env = row["target_env_name"]
        intent_text = row["target_intent"]
        item = reader.get(sid)
        features = item["features"]
        ptype = str(row["position_type"])
        pos = int(row["position_index"])
        h = features[pos].contiguous().to(torch.float32)

        # Decode caption with AV (matched intent) — mirror compare_cf_steer_checkpoints
        out = av.generate(
            h.unsqueeze(0).to(args.device),
            [ptype],
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            target_intent_texts=[intent_text],
        )
        text = out["text"][0]
        steer = encode_texts_with_ar(ar, [text], device=args.device)
        if steer.dim() == 3 and args.sim_placement not in ("image_patch_spatial", "image_patch_strided"):
            steer = steer.mean(dim=1)
        per_sample_text[sid] = text
        per_sample_steer[sid] = steer
        per_sample_meta[sid] = {"task": task, "env": env, "intent": intent_text,
                                "position_type": ptype, "position_index": pos}
        print(f"[probe] {sid}: caption[:80]={text[:80]!r}", flush=True)

    # Build N reseed jobs per sample
    all_jobs: list = []
    all_meta: list = []
    seeds = [s * 17 for s in range(args.n_reseeds)]   # 0,17,34,...,153
    for sid in target_ids:
        meta = per_sample_meta[sid]
        steer = per_sample_steer[sid]
        for k, seed in enumerate(seeds):
            jobs = assemble_jobs(
                rollout_texts=[per_sample_text[sid]],
                steer_vecs=steer,
                target_tasks=[meta["task"]],
                target_env_names=[meta["env"]],
                source_ids=[sid],
                seeds=[seed],
                sim_max_steps=args.sim_max_steps,
                placement=args.sim_placement,
                blend=1.0,
                policy_language_overrides=[meta["intent"]],
                steer_disabled=False,
                strided_k=args.strided_k,
            )
            all_jobs.append(jobs[0])
            all_meta.append({"sid": sid, "seed": seed, "reseed_idx": k})

    print(f"[probe] {len(all_jobs)} jobs total ({len(target_ids)} configs x {args.n_reseeds} reseeds)", flush=True)

    libero_py = str(Path(__file__).resolve().parents[2]
                    / "third_party/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python")
    worker = SimRewardWorker(
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        n_workers=1,
        sim_batch_size=args.sim_batch_size,
        rollout_python=libero_py,
        cache_path=args.sim_cache_path,
    )
    print(f"[probe] worker ready; in-mem cache={len(worker._cache)}", flush=True)
    t0 = time.time()
    results = worker.compute(all_jobs)
    print(f"[probe] all rollouts done ({time.time()-t0:.1f}s)", flush=True)

    # Aggregate per config
    per_config: dict[str, list[float]] = {sid: [] for sid in target_ids}
    rows_out = []
    for meta, res in zip(all_meta, results):
        r_sim = float(res.r_sim) if hasattr(res, "r_sim") else float(res.get("r_sim", float("nan")))
        per_config[meta["sid"]].append(r_sim)
        rows_out.append({"sid": meta["sid"], "seed": meta["seed"], "reseed_idx": meta["reseed_idx"],
                         "r_sim": r_sim})
        print(f"[probe] {meta['sid']} seed={meta['seed']} r_sim={r_sim:.4f}", flush=True)

    per_config_summary = []
    pooled_var = 0.0
    pooled_n = 0
    for sid, vals in per_config.items():
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        per_config_summary.append({"sid": sid, "n": len(vals), "mean": m, "std": s,
                                   "min": min(vals), "max": max(vals)})
        # Pool variance across configs (each config contributes (n-1) df)
        pooled_var += (len(vals) - 1) * (s ** 2)
        pooled_n += (len(vals) - 1)
    pooled_within_std = math.sqrt(pooled_var / pooled_n) if pooled_n > 0 else 0.0

    # Between-config std (from n=100 eval, M_sem arm) for comparison
    BETWEEN_CONFIG_STD = 0.099   # std of M_sem r_sim over n=100 configs

    out = {
        "n_configs": len(target_ids),
        "n_reseeds": args.n_reseeds,
        "per_config": per_config_summary,
        "pooled_within_config_std": pooled_within_std,
        "between_config_std_reference_n100": BETWEEN_CONFIG_STD,
        "variance_explained_by_rollout_noise":
            (pooled_within_std ** 2) / (BETWEEN_CONFIG_STD ** 2) if BETWEEN_CONFIG_STD > 0 else None,
        "rows": rows_out,
        "config": vars(args),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[probe] wrote {args.out_json}", flush=True)
    print()
    print("--- SUMMARY ---")
    print(f"Pooled within-config std (rollout noise):   {pooled_within_std:.4f}")
    print(f"Between-config std (M_sem, n=100 eval):     {BETWEEN_CONFIG_STD:.4f}")
    print(f"Rollout noise variance / between-config variance = {(pooled_within_std**2)/(BETWEEN_CONFIG_STD**2):.2%}")
    for s in per_config_summary:
        print(f"  {s['sid']}: n={s['n']}  mean={s['mean']:.4f}  std={s['std']:.4f}  [{s['min']:.3f}, {s['max']:.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
