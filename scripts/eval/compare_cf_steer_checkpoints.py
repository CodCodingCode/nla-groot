#!/usr/bin/env python
"""Compare SFT vs GRPO AV on counterfactual LIBERO steer rollouts.

For each sampled CF pair:
  activation h  ->  AV.generate(h | target_intent)  ->  AR(text)  ->  steer_h
  ->  short LIBERO rollout  ->  predicate / r_sim on target_task

This is the same path sim-GRPO trains on; use it to see if GRPO moved steerability
vs the V5 SFT AV baseline.

Usage::

    # Steer server must be up on --policy-port (same AR as SFT).
    PYTHONPATH=src .venv/bin/python scripts/eval/compare_cf_steer_checkpoints.py \\
        --sft-dir data/sft/libero_4suite_v5_base_qwen \\
        --grpo-av-dir data/grpo/libero_4suite_v5_sim_grpo_pilot/av \\
        --pairs-path data/grpo/libero_goal_counterfactual_pairs.jsonl \\
        --activations-root data/activations/libero_4suite_v4_combined \\
        --n-samples 8 \\
        --out-json data/eval/cf_steer_sft_vs_grpo_pilot.json

Requires ``--grpo-av-dir`` to contain a saved GRPO policy (``adapter_model.safetensors``, etc.).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
ROLLOUT_SCRIPT = ROOT / "src/nla/eval/steerability/rollout.py"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sft-dir", required=True, help="V5 SFT run (ar/ + baseline av/).")
    p.add_argument("--grpo-av-dir", required=True, help="GRPO-trained av/ directory.")
    p.add_argument("--pairs-path", required=True, help="CF pairs JSONL.")
    p.add_argument("--activations-root", required=True)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--position-type", default="image_patch")
    p.add_argument("--sim-max-steps", type=int, default=100)
    p.add_argument("--sim-placement", default="image_patch")
    p.add_argument("--sim-blend", type=float, default=1.0)
    p.add_argument("--policy-host", default="localhost")
    p.add_argument("--policy-port", type=int, default=5556)
    p.add_argument("--sim-rollout-python", default=None)
    p.add_argument("--sim-batch-size", type=int, default=1,
                   help="Rollouts per batched subprocess. ``1`` = legacy "
                        "(one subprocess per rollout, dominated by cold-start "
                        "imports). ``>=2`` uses batched_rollout.py so the "
                        "GR00T policy server can batch inference across envs; "
                        "yields ~3-5x speedup at sim-batch-size=4. The server "
                        "must support ``get_action_batch`` (run_gr00t_server_"
                        "nla_steer.py exposes this by default).")
    p.add_argument("--sim-n-workers", type=int, default=None,
                   help="Parallel worker processes inside SimRewardWorker. "
                        "Default: auto (1 if sim-batch-size>=2 to avoid policy "
                        "server contention; min(4, total_jobs) otherwise).")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="AV decode temperature (0=greedy).")
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-json", required=True)
    p.add_argument("--video-dir", default=None,
                   help="If set, write rollout.mp4 per sample under this tree.")
    p.add_argument("--conditions", default="sft_av,grpo_av",
                   help="Comma-separated: sft_av, grpo_av.")
    p.add_argument("--intent-arms", default="matched",
                   help="Comma-separated intent arms: matched (target_intent), "
                        "mismatched_source (source_intent on same h). "
                        "Default 'matched' preserves legacy behavior; "
                        "use 'matched,mismatched_source' for the semantic_gap "
                        "publishable control.")
    p.add_argument("--causal-arms", default="semantic",
                   help="Comma-separated causal arms applied to the AR(text) "
                        "vector: semantic (real ĥ at trained placement), "
                        "matched_null (Gaussian draw rescaled to ||ĥ||), "
                        "wrong_placement (real ĥ at --wrong-placement instead "
                        "of --sim-placement). Default 'semantic' = legacy.")
    p.add_argument("--wrong-placement", default="last_text",
                   help="Placement used by the wrong_placement causal arm "
                        "(default last_text). Must differ from --sim-placement.")
    p.add_argument("--reuse-pairs-json", default=None,
                   help="Use sample list from a prior compare JSON (same rows).")
    p.add_argument("--only-source-id", default=None,
                   help="Run only this activation id (e.g. the predicate hit).")
    p.add_argument("--sim-timeout-s", type=float, default=300.0)
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned samples only.")
    # Leakage / determinism guards. Held-out eval should always set these.
    p.add_argument("--exclude-ids-path", default=None,
                   help="JSON manifest of train example_ids "
                        "(from build_grpo_cf_eval_manifest.py *_train_manifest.json). "
                        "Pair rows whose source_example_id appears here are dropped "
                        "before sampling.")
    p.add_argument("--require-held-out", action="store_true",
                   help="Fail with rc=3 if any selected sample's source_example_id "
                        "appears in --exclude-ids-path; ensures no train leakage.")
    p.add_argument("--deterministic-order", action="store_true",
                   help="Iterate pairs in file order (no shuffle). Combined with "
                        "--require-held-out gives a frozen, reproducible eval slice.")
    p.add_argument("--forbid-sim-cache", action="store_true",
                   help="Assert that the sim reward worker is constructed with "
                        "cache_path=None (no train cache reuse during eval).")
    return p


def _run_rollout_video(
    *,
    job,
    rollout_python: str,
    policy_host: str,
    policy_port: int,
    output_dir: Path,
    timeout_s: float,
) -> dict:
    """LIBERO rollout with MP4 capture (no --no-frames)."""
    from nla.eval.steerability.predicates import tracked_bodies_for

    output_dir.mkdir(parents=True, exist_ok=True)
    steer_path = output_dir / "steer_h.npy"
    np.save(steer_path, job.steer_h.astype(np.float32, copy=False))
    bodies = tracked_bodies_for(job.target_task)

    cmd = [
        rollout_python,
        str(ROLLOUT_SCRIPT),
        "--env-name", job.env_name,
        "--seed", str(int(job.seed)),
        "--policy-host", policy_host,
        "--policy-port", str(int(policy_port)),
        "--target-task", job.target_task,
        "--steer-h-path", str(steer_path),
        "--steer-placement", job.placement,
        "--steer-blend", f"{job.blend:.3f}",
        "--max-episode-steps", str(int(job.sim_max_steps)),
        "--output-dir", str(output_dir),
        "--early-stop-on-success",
        "--fps", "20",
        "--steps-per-render", "1",
    ]
    for b in bodies:
        cmd.extend(["--tracked-bodies", b])

    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "osmesa")
    env.setdefault("PYOPENGL_PLATFORM", "osmesa")
    completed = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout_s, env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"rollout failed rc={completed.returncode}; stderr={completed.stderr[-800:]!r}"
        )
    text = completed.stdout.strip()
    idx = text.find("{")
    summary = json.loads(text[idx:] if idx >= 0 else text)
    mp4 = output_dir / "rollout.mp4"
    summary["video_path"] = str(mp4) if mp4.is_file() else None
    return summary


def _load_exclude_ids(path: Path | None) -> set[str]:
    """Return the set of example_ids in a train manifest (or empty)."""
    if path is None:
        return set()
    obj = json.loads(Path(path).read_text())
    ids = obj.get("example_ids")
    if not isinstance(ids, list):
        raise ValueError(
            f"--exclude-ids-path {path} missing 'example_ids' list; "
            "expected output of build_grpo_cf_eval_manifest.py."
        )
    return {str(x) for x in ids}


def _load_pairs(
    path: Path,
    n: int,
    rng: random.Random,
    *,
    exclude_ids: set[str] | None = None,
    deterministic_order: bool = False,
) -> list[dict]:
    rows: list[dict] = []
    n_dropped_exclude = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not (obj.get("target_task") and obj.get("target_env_name")):
                continue
            sid = str(obj.get("source_example_id") or "")
            if exclude_ids and sid in exclude_ids:
                n_dropped_exclude += 1
                continue
            rows.append(obj)
    if not rows:
        raise RuntimeError(f"No valid CF rows in {path}")
    if n_dropped_exclude:
        print(
            f"[load_pairs] dropped {n_dropped_exclude} rows whose "
            "source_example_id appeared in --exclude-ids-path"
        )
    if not deterministic_order:
        rng.shuffle(rows)
    return rows[: min(n, len(rows))]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    sft_dir = Path(args.sft_dir)
    grpo_av = Path(args.grpo_av_dir)
    pairs_path = Path(args.pairs_path)
    out_json = Path(args.out_json)

    # Validate arm CLI early -- before any heavy load (ActivationShardReader,
    # AV/AR weights, sim worker) so bad flags fail fast in CI / smoke tests.
    intent_arms = [a.strip() for a in args.intent_arms.split(",") if a.strip()]
    _VALID_INTENT_ARMS = {"matched", "mismatched_source"}
    for arm in intent_arms:
        if arm not in _VALID_INTENT_ARMS:
            print(
                f"FATAL: --intent-arms entry {arm!r} not in {sorted(_VALID_INTENT_ARMS)}",
                file=sys.stderr,
            )
            return 2
    causal_arms = [a.strip() for a in args.causal_arms.split(",") if a.strip()]
    _VALID_CAUSAL_ARMS = {"semantic", "matched_null", "wrong_placement"}
    for arm in causal_arms:
        if arm not in _VALID_CAUSAL_ARMS:
            print(
                f"FATAL: --causal-arms entry {arm!r} not in "
                f"{sorted(_VALID_CAUSAL_ARMS)}",
                file=sys.stderr,
            )
            return 2
    if "wrong_placement" in causal_arms and args.wrong_placement == args.sim_placement:
        print(
            f"FATAL: --wrong-placement ({args.wrong_placement}) must differ "
            f"from --sim-placement ({args.sim_placement})",
            file=sys.stderr,
        )
        return 2

    for label, p in [
        ("sft-dir", sft_dir),
        ("pairs", pairs_path),
    ]:
        if not p.exists():
            print(f"FATAL: {label} missing: {p}", file=sys.stderr)
            return 2

    for name, av_dir in [("sft", sft_dir / "av"), ("grpo", grpo_av)]:
        if not (av_dir / "av_config.json").is_file():
            print(
                f"FATAL: {name} av checkpoint incomplete: {av_dir}\n"
                "  (wait for GRPO save at step 100 / final, or pass a valid --grpo-av-dir)",
                file=sys.stderr,
            )
            return 2
    if not (sft_dir / "ar" / "ar_config.json").is_file():
        print(f"FATAL: missing AR at {sft_dir}/ar", file=sys.stderr)
        return 2

    libero_py = args.sim_rollout_python
    if libero_py is None:
        libero_py = str(
            ROOT / "third_party/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"
        )

    exclude_ids = _load_exclude_ids(
        Path(args.exclude_ids_path) if args.exclude_ids_path else None
    )

    if args.reuse_pairs_json:
        prev = json.loads(Path(args.reuse_pairs_json).read_text())
        samples = [
            {
                "source_example_id": s["source_example_id"],
                "target_intent": s["target_intent"],
                "target_task": s["target_task"],
                "target_env_name": s["target_env_name"],
                "position_index": s.get("position_index"),
                "position_type": s.get("position_type"),
                "source_intent": s.get("source_intent"),
                "source_task": s.get("source_task"),
                "is_counterfactual": s.get("is_counterfactual"),
            }
            for s in prev.get("samples", [])
        ]
        if args.n_samples < len(samples):
            samples = samples[: args.n_samples]
    else:
        rng = random.Random(args.seed)
        samples = _load_pairs(
            pairs_path,
            args.n_samples,
            rng,
            exclude_ids=exclude_ids,
            deterministic_order=args.deterministic_order,
        )

    if args.require_held_out and exclude_ids:
        leaks = [s["source_example_id"] for s in samples
                 if s["source_example_id"] in exclude_ids]
        if leaks:
            print(
                f"FATAL: --require-held-out: {len(leaks)} sample(s) appear in "
                f"--exclude-ids-path (e.g. {leaks[:3]})",
                file=sys.stderr,
            )
            return 3

    if args.only_source_id:
        samples = [s for s in samples if s["source_example_id"] == args.only_source_id]
        if not samples:
            # Look up one row from pairs file.
            with pairs_path.open() as f:
                for line in f:
                    obj = json.loads(line)
                    if obj.get("source_example_id") == args.only_source_id:
                        samples = [obj]
                        break
        if not samples:
            print(f"FATAL: no pair for {args.only_source_id}", file=sys.stderr)
            return 2

    print(f"CF steer compare: {len(samples)} samples")
    for i, row in enumerate(samples):
        print(f"  [{i}] {row['source_example_id']} -> {row['target_task']}")
    if args.dry_run:
        return 0

    from nla.extraction.storage import ActivationShardReader
    from nla.steering.null_controls import matched_null_vec
    from nla.training.checkpoint import load_ar_from_sft, load_av_from_sft
    from nla.training.sim_reward import SimRewardWorker, assemble_jobs, encode_texts_with_ar

    def _make_record_key(cond_name: str, intent_arm: str, causal_arm: str) -> str:
        # Backward-compatible key: legacy clients reading "sft_av" / "grpo_av"
        # keep working when only the default arms are in play.
        if intent_arm == "matched" and causal_arm == "semantic":
            return cond_name
        if causal_arm == "semantic":
            return f"{cond_name}__{intent_arm}"
        if intent_arm == "matched":
            return f"{cond_name}__{causal_arm}"
        return f"{cond_name}__{intent_arm}__{causal_arm}"

    def _apply_causal_arm(
        steer_real: torch.Tensor,
        *,
        sample_index: int,
        seed_base: int,
        causal_arm: str,
        trained_placement: str,
        wrong_placement: str,
    ) -> tuple[torch.Tensor, str]:
        """Return (steer_vec, placement) for a given causal arm.

        ``semantic``         keeps the AR(text) vector and trained placement.
        ``matched_null``     replaces the vector with a Gaussian rescaled to
                             the same L2 norm; placement stays trained.
        ``wrong_placement``  keeps the AR vector but moves it to a different
                             token role.
        """
        if causal_arm == "semantic":
            return steer_real, trained_placement
        if causal_arm == "matched_null":
            # Deterministic per (seed_base, sample_index) so re-runs are
            # reproducible. Re-norm to the trained-placement vector's L2.
            null = torch.stack([
                matched_null_vec(
                    steer_real[b], seed=(seed_base + sample_index) * 31 + b,
                )
                for b in range(steer_real.shape[0])
            ], dim=0)
            return null.to(steer_real.device), trained_placement
        if causal_arm == "wrong_placement":
            return steer_real, wrong_placement
        raise ValueError(f"unknown causal arm: {causal_arm!r}")

    reader = ActivationShardReader(args.activations_root)
    ar = load_ar_from_sft(sft_dir / "ar", device=args.device, freeze=True)
    av_sft = load_av_from_sft(sft_dir / "av", device=args.device, freeze=True)
    av_grpo = load_av_from_sft(grpo_av, device=args.device, freeze=True)

    cond_list = [c.strip() for c in args.conditions.split(",") if c.strip()]
    av_by_cond = {"sft_av": av_sft, "grpo_av": av_grpo}

    worker = None
    if args.video_dir is None:
        # cache_path=None is critical for publishable eval: never reuse train
        # cache values, even if the key happens to collide. --forbid-sim-cache
        # makes the assumption explicit (script never passes a cache path
        # anyway, but flag-gating documents intent in the saved config).
        cache_path = None
        if args.forbid_sim_cache and cache_path is not None:
            print("FATAL: --forbid-sim-cache violated", file=sys.stderr)
            return 4
        total_jobs = max(
            1, len(samples) * len(cond_list) * len(intent_arms) * len(causal_arms),
        )
        sim_batch_size = max(1, int(args.sim_batch_size))
        if args.sim_n_workers is None:
            # When batching is on, a single worker driving the batched
            # subprocess fully utilizes the policy server. Spawning multiple
            # workers without spinning up extra GR00T servers just serializes
            # on the shared server and burns subprocess overhead.
            n_workers = 1 if sim_batch_size >= 2 else min(4, total_jobs)
        else:
            n_workers = max(1, int(args.sim_n_workers))
        print(
            f"[compare] SimRewardWorker config: sim_batch_size={sim_batch_size} "
            f"n_workers={n_workers} total_jobs={total_jobs}",
            flush=True,
        )
        worker = SimRewardWorker(
            policy_host=args.policy_host,
            policy_port=args.policy_port,
            n_workers=n_workers,
            sim_batch_size=sim_batch_size,
            rollout_python=libero_py,
            cache_path=cache_path,
        )
    video_root = Path(args.video_dir) if args.video_dir else None

    results: list[dict] = []
    for i, row in enumerate(samples):
        sid = row["source_example_id"]
        intent = row["target_intent"]
        task = row["target_task"]
        env = row["target_env_name"]
        item = reader.get(sid)
        features = item["features"]
        if row.get("position_index") is not None and row.get("position_type"):
            pos = int(row["position_index"])
            ptype = str(row["position_type"])
        else:
            from nla.training.dataset import TokenPositionSampler
            sampler = TokenPositionSampler(seed=args.seed + i)
            ptype, pos = sampler.sample(item["attention_mask"], item["image_mask"])
        if pos >= features.shape[0]:
            raise RuntimeError(
                f"position {pos} >= seq_len {features.shape[0]} for {sid}"
            )
        h = features[pos].contiguous().to(torch.float32)

        # `scoring` block makes the metric semantics explicit so reviewers and
        # downstream scorecard scripts never confuse xyz-heuristic predicate
        # with native LIBERO BDDL success. env_matches_scored_task=False is
        # the cross-scene CF case; predicate is then a steered-task heuristic.
        scored_env = f"libero_sim/{task}" if task else ""
        record: dict = {
            "source_example_id": sid,
            "position_type": ptype,
            "position_index": pos,
            "target_task": task,
            "target_env_name": env,
            "target_intent": intent[:200],
            "source_intent": (str(row.get("source_intent") or "")[:200]) or None,
            "source_task": row.get("source_task") or None,
            "is_counterfactual": bool(row.get("is_counterfactual"))
                if row.get("is_counterfactual") is not None else None,
            "scoring": {
                "scored_task": task,
                "loaded_env": env,
                "env_matches_scored_task": bool(env == scored_env),
                "source_suite": sid.split("__", 1)[0] if "__" in sid else None,
                "predicate_kind": "xyz_heuristic_on_target_task",
            },
            "conditions": {},
        }

        # Map intent_arm -> the text we condition AV on. Returns None when
        # an arm cannot run for this row (e.g. mismatched_source needs a
        # non-empty source_intent); the arm is then recorded as skipped.
        def _intent_text_for_arm(arm: str) -> str | None:
            if arm == "matched":
                return intent
            if arm == "mismatched_source":
                si = (row.get("source_intent") or "").strip()
                return si or None
            raise ValueError(f"unknown intent arm: {arm!r}")

        for cond_name in cond_list:
            av = av_by_cond.get(cond_name)
            if av is None:
                raise ValueError(f"unknown condition {cond_name}")
            for intent_arm in intent_arms:
                intent_text = _intent_text_for_arm(intent_arm)
                if intent_text is None:
                    # No source intent for this row; mark every causal arm as
                    # skipped so downstream aggregation can drop them cleanly.
                    for causal_arm in causal_arms:
                        record_key = _make_record_key(cond_name, intent_arm, causal_arm)
                        record["conditions"][record_key] = {
                            "intent_arm": intent_arm,
                            "causal_arm": causal_arm,
                            "skipped_reason": f"no_intent_for_arm:{intent_arm}",
                            "predicate": 0.0,
                            "r_sim": 0.0,
                            "success_xyz_predicate": 0,
                            "success_any": False,
                            "success_bddl_native": False,
                            "error": None,
                        }
                        print(
                            f"  [{i}] {record_key}: skipped (no intent for arm)",
                            flush=True,
                        )
                    continue
                with torch.no_grad():
                    out = av.generate(
                        h.unsqueeze(0).to(args.device),
                        [ptype],
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=1.0,
                        do_sample=args.temperature > 0,
                        target_intent_texts=[intent_text],
                    )
                text = out["text"][0]
                steer_real = encode_texts_with_ar(ar, [text], device=args.device)
                # Shared LIBERO seed across all conditions and arms for a
                # given sample so SFT vs GRPO see the same env RNG.
                seed = args.seed + i * 17
                for causal_arm in causal_arms:
                    record_key = _make_record_key(cond_name, intent_arm, causal_arm)
                    steer_vec_for_arm, placement_for_arm = _apply_causal_arm(
                        steer_real,
                        sample_index=i,
                        seed_base=args.seed,
                        causal_arm=causal_arm,
                        trained_placement=args.sim_placement,
                        wrong_placement=args.wrong_placement,
                    )
                    t0 = time.time()
                    jobs = assemble_jobs(
                        rollout_texts=[text],
                        steer_vecs=steer_vec_for_arm,
                        target_tasks=[task],
                        target_env_names=[env],
                        source_ids=[sid],
                        seeds=[seed],
                        sim_max_steps=args.sim_max_steps,
                        placement=placement_for_arm,
                        blend=args.sim_blend,
                    )
                    job = jobs[0]
                    err = None
                    video_path = None
                    if video_root is not None:
                        slug = f"{i:02d}_{sid}__{record_key}__{task}"
                        out_dir = video_root / slug
                        try:
                            summ = _run_rollout_video(
                                job=job,
                                rollout_python=libero_py,
                                policy_host=args.policy_host,
                                policy_port=args.policy_port,
                                output_dir=out_dir,
                                timeout_s=args.sim_timeout_s,
                            )
                            pred = float((summ.get("sim_score_breakdown") or {}).get("predicate", 0))
                            r_sim = float(summ.get("r_sim") or 0)
                            video_path = summ.get("video_path")
                            early = bool(summ.get("early_stopped", False))
                            n_steps = int(summ.get("n_steps", 0))
                            succ = bool(summ.get("success_any", False))
                        except Exception as e:
                            pred, r_sim, early, n_steps, succ = 0.0, 0.0, False, 0, False
                            err = str(e)
                        (out_dir / "caption.txt").write_text(text)
                        # Both labeled aliases and legacy keys; downstream
                        # scorecards should prefer the labeled forms
                        # (`success_xyz_predicate`, `success_bddl_native`).
                        record["conditions"][record_key] = {
                            "intent_arm": intent_arm,
                            "causal_arm": causal_arm,
                            "av_condition": cond_name,
                            "placement": placement_for_arm,
                            "text_preview": text[:500],
                            "r_sim": r_sim,
                            "predicate": pred,
                            "success_xyz_predicate": int(pred > 0),
                            "success_any": succ,
                            "success_bddl_native": bool(succ),
                            "early_stopped": early,
                            "n_steps": n_steps,
                            "elapsed_s": round(time.time() - t0, 1),
                            "error": err,
                            "video_path": video_path,
                            "rollout_dir": str(out_dir),
                        }
                    else:
                        assert worker is not None
                        sim_res = worker.compute(jobs)[0]
                        record["conditions"][record_key] = {
                            "intent_arm": intent_arm,
                            "causal_arm": causal_arm,
                            "av_condition": cond_name,
                            "placement": placement_for_arm,
                            "text_preview": text[:500],
                            "r_sim": sim_res.r_sim,
                            "predicate": sim_res.predicate,
                            "success_xyz_predicate": int(sim_res.predicate > 0),
                            "success_any": sim_res.success_any,
                            "success_bddl_native": bool(sim_res.success_any),
                            "early_stopped": sim_res.early_stopped,
                            "n_steps": sim_res.n_steps,
                            "elapsed_s": round(time.time() - t0, 1),
                            "error": sim_res.error,
                        }
                    c = record["conditions"][record_key]
                    print(
                        f"  [{i}] {record_key}: pred={c['predicate']:.0f} "
                        f"r_sim={c['r_sim']:.2f} video={c.get('video_path')} "
                        f"err={c['error']}",
                        flush=True,
                    )

        results.append(record)

    def _rate_only(rows: list[dict], key: str, metric: str) -> float:
        vals = [r["conditions"][key][metric] for r in rows
                if key in r["conditions"] and r["conditions"][key].get("error") is None
                and "skipped_reason" not in r["conditions"][key]]
        if not vals:
            return 0.0
        if metric == "predicate":
            return sum(1 for v in vals if v > 0) / len(vals)
        return sum(bool(v) for v in vals) / len(vals)

    summary = {
        "n": len(results),
        "config": vars(args),
        "exclude_ids_count": len(exclude_ids),
        "intent_arms": intent_arms,
        "causal_arms": causal_arms,
        "samples": results,
    }
    for cn in cond_list:
        for intent_arm in intent_arms:
            for causal_arm in causal_arms:
                key = _make_record_key(cn, intent_arm, causal_arm)
                pred_rate = _rate_only(results, key, "predicate")
                bddl_rate = _rate_only(results, key, "success_any")
                r_sim_vals = [
                    r["conditions"][key]["r_sim"] for r in results
                    if key in r["conditions"]
                    and r["conditions"][key].get("error") is None
                    and "skipped_reason" not in r["conditions"][key]
                ]
                mean_r_sim = sum(r_sim_vals) / max(1, len(r_sim_vals))
                if intent_arm == "matched" and causal_arm == "semantic":
                    # Legacy keys (kept for tooling that already reads them).
                    summary[f"{cn}_predicate_rate"] = pred_rate
                    summary[f"{cn}_success_any_rate"] = bddl_rate
                    summary[f"mean_r_sim_{cn}"] = mean_r_sim
                summary[f"{key}_predicate_rate"] = pred_rate
                summary[f"{key}_success_xyz_predicate_rate"] = pred_rate
                summary[f"{key}_success_bddl_native_rate"] = bddl_rate
                summary[f"{key}_mean_r_sim"] = mean_r_sim
                summary[f"{key}_n_active"] = len(r_sim_vals)

    # Semantic gap per condition: matched − mismatched_source predicate rate
    # on the *semantic* causal arm (so the comparison isn't muddled by null
    # vectors). Publishable claim "language is causal" needs gap > 0.
    if "mismatched_source" in intent_arms and "matched" in intent_arms:
        for cn in cond_list:
            mk = _make_record_key(cn, "matched", "semantic")
            wk = _make_record_key(cn, "mismatched_source", "semantic")
            mr = summary.get(f"{mk}_predicate_rate", 0.0)
            wr = summary.get(f"{wk}_predicate_rate", 0.0)
            summary[f"{cn}_semantic_gap_predicate"] = mr - wr
            summary[f"{cn}_paired_semantic_wins"] = sum(
                1 for r in results
                if (r["conditions"].get(mk, {}).get("predicate", 0) or 0) > 0
                and (r["conditions"].get(wk, {}).get("predicate", 0) or 0) == 0
            )

    # Causal specificity per condition: semantic − matched_null (norm-matched
    # noise) and semantic − wrong_placement (site specificity). Both should
    # be > 0 for "the steer is doing semantic work at the right token role".
    if "matched_null" in causal_arms:
        for cn in cond_list:
            sk = _make_record_key(cn, "matched", "semantic")
            nk = _make_record_key(cn, "matched", "matched_null")
            summary[f"{cn}_causal_specificity_predicate"] = (
                summary.get(f"{sk}_predicate_rate", 0.0)
                - summary.get(f"{nk}_predicate_rate", 0.0)
            )
    if "wrong_placement" in causal_arms:
        for cn in cond_list:
            sk = _make_record_key(cn, "matched", "semantic")
            wk = _make_record_key(cn, "matched", "wrong_placement")
            summary[f"{cn}_placement_specificity_predicate"] = (
                summary.get(f"{sk}_predicate_rate", 0.0)
                - summary.get(f"{wk}_predicate_rate", 0.0)
            )

    # Headline beat-SFT deltas: GRPO − SFT predicate rate (the V2 success
    # criterion). Reported separately so scorecards don't have to recompute.
    if "sft_av" in cond_list and "grpo_av" in cond_list:
        summary["delta_predicate_rate_grpo_minus_sft"] = (
            summary["grpo_av_predicate_rate"] - summary["sft_av_predicate_rate"]
        )
        summary["delta_success_bddl_native_rate_grpo_minus_sft"] = (
            summary["grpo_av_success_any_rate"]
            - summary["sft_av_success_any_rate"]
        )
        summary["paired_wins_grpo_predicate"] = sum(
            1 for r in results
            if (r["conditions"].get("grpo_av", {}).get("predicate", 0) or 0) > 0
            and (r["conditions"].get("sft_av", {}).get("predicate", 0) or 0) == 0
        )
        summary["paired_losses_grpo_predicate"] = sum(
            1 for r in results
            if (r["conditions"].get("grpo_av", {}).get("predicate", 0) or 0) == 0
            and (r["conditions"].get("sft_av", {}).get("predicate", 0) or 0) > 0
        )
    if video_root is not None:
        summary["video_dir"] = str(video_root)
        vids = []
        for r in results:
            for cn, c in r["conditions"].items():
                if c.get("video_path"):
                    vids.append(c["video_path"])
        summary["videos"] = vids
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_json}")
    for cn in cond_list:
        for intent_arm in intent_arms:
            for causal_arm in causal_arms:
                key = _make_record_key(cn, intent_arm, causal_arm)
                pkey = f"{key}_predicate_rate"
                if pkey in summary:
                    print(
                        f"  {key} predicate rate "
                        f"(xyz-heuristic on target_task): "
                        f"{summary[pkey]:.2%}"
                    )
        gap_key = f"{cn}_semantic_gap_predicate"
        if gap_key in summary:
            print(
                f"  {cn} semantic_gap_predicate (matched - mismatched): "
                f"{summary[gap_key]:+.2%}"
            )
        cs_key = f"{cn}_causal_specificity_predicate"
        if cs_key in summary:
            print(
                f"  {cn} causal_specificity_predicate (semantic - matched_null): "
                f"{summary[cs_key]:+.2%}"
            )
        ps_key = f"{cn}_placement_specificity_predicate"
        if ps_key in summary:
            print(
                f"  {cn} placement_specificity_predicate "
                f"(semantic - wrong_placement): "
                f"{summary[ps_key]:+.2%}"
            )
    if "delta_predicate_rate_grpo_minus_sft" in summary:
        print(
            f"  delta_predicate_rate (GRPO - SFT, matched/semantic): "
            f"{summary['delta_predicate_rate_grpo_minus_sft']:+.2%}"
        )
    if summary.get("videos"):
        print(f"  videos ({len(summary['videos'])}):")
        for v in summary["videos"]:
            print(f"    {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
