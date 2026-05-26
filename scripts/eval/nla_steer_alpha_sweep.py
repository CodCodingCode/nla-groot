#!/usr/bin/env python
"""Dose (alpha) sweep over closed-loop CF steer A/B — Stage 0 diagnostic.

Wraps ``compare_cf_steer_checkpoints.py`` and runs it once per alpha value,
sharing one already-up steer server. The point is to distinguish two
hypotheses for the current Δ_cw=0 observation on image_patch steers:

  (a) dose-miscalibration  → at some alpha, matched intent beats mismatched
      by ≥+5pp (Δ_cw > 0). Codec is fine; we just hit the policy too hard or
      too soft.
  (b) codec failure        → Δ_cw stays in [-2pp, +2pp] across every alpha.
      The AR vector itself is not semantic on the vision slots; no dose
      rescaling will save it. Proceed to Stage 2+ retrain.

Required: the steer server (``launch_steer_server.sh``) is already running
on ``--policy-host:--policy-port`` and exposes ``get_action_batch``. This
wrapper only changes the per-call ``alpha_scale`` passed to compare; the
server itself stays at its bootstrap alpha and is not reloaded between
conditions, so the sweep is fast.

Sanity arms:
  alpha_scale=0.0  ⇒ steer vector is the zero vector, succ should match
                     baseline (no-steer). If not, server is misconfigured.
  alpha_scale≫1    ⇒ steer vector goes OOD, succ should collapse to ~0.
                     If not, the steer hook never fired.

Example::

    PYTHONPATH=src python scripts/eval/nla_steer_alpha_sweep.py \\
        --sft-dir          data/sft/libero_4suite_v5_base_qwen \\
        --grpo-av-dir      data/grpo/libero_4suite_v5_sim_grpo_pilot/av \\
        --pairs-path       data/grpo/libero_goal_counterfactual_pairs.jsonl \\
        --activations-root data/activations/libero_4suite_v4_combined \\
        --alpha-scales     0.0,0.25,0.5,0.75,1.0,1.5,2.0 \\
        --n-samples        12 \\
        --intent-arms      matched,mismatched_source \\
        --causal-arms      semantic,no_steer \\
        --sim-placement    image_patch_all \\
        --policy-port      5555 \\
        --out-dir          runs/alpha_sweep/$(date +%Y%m%d_%H%M)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPARE_SCRIPT = ROOT / "scripts/eval/compare_cf_steer_checkpoints.py"


def _parse_float_list(spec: str) -> list[float]:
    return [float(x) for x in spec.split(",") if x.strip()]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # Sweep-specific
    p.add_argument("--alpha-scales", default="0.0,0.25,0.5,0.75,1.0,1.5,2.0",
                   help="Comma-separated multiplicative dose scales. 1.0 = "
                        "trained alpha; 0.0 is the zero-steer sanity check.")
    p.add_argument("--out-dir", required=True,
                   help="Output directory. Per-alpha JSONs land here as "
                        "alpha_<scale>.json plus a summary.json aggregating "
                        "Δ_cw, predicate rates, and run config.")
    # Pass-through to compare_cf_steer_checkpoints.py
    p.add_argument("--sft-dir", required=True)
    p.add_argument("--grpo-av-dir", required=True)
    p.add_argument("--pairs-path", required=True)
    p.add_argument("--activations-root", required=True)
    p.add_argument("--n-samples", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--position-type", default="image_patch")
    p.add_argument("--sim-max-steps", type=int, default=100)
    p.add_argument("--sim-placement", default="image_patch_all",
                   help="Default image_patch_all so the dose actually lands "
                        "across all vision slots; if it only lands on one "
                        "patch index it'll be averaged-away by the other "
                        "vision tokens.")
    p.add_argument("--sim-blend", type=float, default=1.0)
    p.add_argument("--policy-host", default="localhost")
    p.add_argument("--policy-port", type=int, default=5555)
    p.add_argument("--sim-rollout-python", default=None)
    p.add_argument("--sim-batch-size", type=int, default=4)
    p.add_argument("--sim-n-workers", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--device", default="cuda")
    p.add_argument("--conditions", default="sft_av",
                   help="Default just sft_av (one model per sweep); add "
                        "grpo_av only if you also want to dose-sweep GRPO.")
    p.add_argument("--intent-arms", default="matched,mismatched_source",
                   help="Both arms are needed for Δ_cw = matched − "
                        "mismatched_source predicate rate.")
    p.add_argument("--causal-arms", default="semantic,no_steer",
                   help="Default includes no_steer so each alpha row also "
                        "reports steer_lift (semantic − no_steer).")
    p.add_argument("--wrong-placement", default="last_text")
    p.add_argument("--eval-protocol", default="language_swap",
                   choices=["legacy", "language_swap"])
    p.add_argument("--exclude-ids-path", default=None)
    p.add_argument("--require-held-out", action="store_true")
    p.add_argument("--deterministic-order", action="store_true")
    p.add_argument("--forbid-sim-cache", action="store_true")
    p.add_argument("--require-distinct-intents", action="store_true",
                   help="Drop pair rows whose source_intent==target_intent or "
                        "source_task==target_task before sampling. Required "
                        "when the pairs file mixes CF and non-CF rows (e.g. "
                        "libero_*_counterfactual_pairs.jsonl is ~50%% non-CF "
                        "in-distribution baselines) — otherwise matched / "
                        "mismatched_source intent arms collapse and Δ_cw=0 "
                        "by construction.")
    p.add_argument("--sim-timeout-s", type=float, default=300.0)
    p.add_argument("--py-bin", default=sys.executable,
                   help="Python interpreter for the compare subprocess. "
                        "Default: same as this script.")
    return p


def _run_one_alpha(args: argparse.Namespace, alpha: float, out_json: Path) -> dict:
    """Invoke compare_cf_steer_checkpoints.py for one alpha; return the summary."""
    cmd = [
        args.py_bin, str(COMPARE_SCRIPT),
        "--sft-dir", args.sft_dir,
        "--grpo-av-dir", args.grpo_av_dir,
        "--pairs-path", args.pairs_path,
        "--activations-root", args.activations_root,
        "--n-samples", str(args.n_samples),
        "--seed", str(args.seed),
        "--position-type", args.position_type,
        "--sim-max-steps", str(args.sim_max_steps),
        "--sim-placement", args.sim_placement,
        "--sim-blend", f"{args.sim_blend:.3f}",
        "--alpha-scale", f"{alpha:.4f}",
        "--policy-host", args.policy_host,
        "--policy-port", str(args.policy_port),
        "--sim-batch-size", str(args.sim_batch_size),
        "--temperature", f"{args.temperature:.3f}",
        "--max-new-tokens", str(args.max_new_tokens),
        "--device", args.device,
        "--conditions", args.conditions,
        "--intent-arms", args.intent_arms,
        "--causal-arms", args.causal_arms,
        "--wrong-placement", args.wrong_placement,
        "--eval-protocol", args.eval_protocol,
        "--sim-timeout-s", f"{args.sim_timeout_s:.0f}",
        "--out-json", str(out_json),
    ]
    if args.sim_rollout_python:
        cmd.extend(["--sim-rollout-python", args.sim_rollout_python])
    if args.sim_n_workers is not None:
        cmd.extend(["--sim-n-workers", str(args.sim_n_workers)])
    if args.exclude_ids_path:
        cmd.extend(["--exclude-ids-path", args.exclude_ids_path])
    if args.require_held_out:
        cmd.append("--require-held-out")
    if args.deterministic_order:
        cmd.append("--deterministic-order")
    if args.forbid_sim_cache:
        cmd.append("--forbid-sim-cache")
    if args.require_distinct_intents:
        cmd.append("--require-distinct-intents")

    print(f"\n=== alpha={alpha:.3f} -> {out_json.name} ===", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, env=None).returncode
    elapsed = time.time() - t0
    if rc != 0:
        print(f"  compare returned rc={rc} (alpha={alpha})", file=sys.stderr)
        return {"alpha_scale": alpha, "error": f"rc={rc}", "elapsed_s": elapsed}
    if not out_json.is_file():
        return {"alpha_scale": alpha, "error": "out_json not written", "elapsed_s": elapsed}
    summ = json.loads(out_json.read_text())
    summ["alpha_scale"] = alpha
    summ["elapsed_s"] = elapsed
    return summ


def _extract_row(summary: dict, cond: str) -> dict:
    """Pull the rows that matter for the dose-sweep decision.

    Handles both the legacy short-form keys ({cond}_predicate_rate for the
    default matched/semantic arm) and the explicit __{intent}__{causal}
    forms emitted for non-default arms by compare_cf_steer_checkpoints.py's
    _make_record_key.
    """
    matched_semantic_keys = (
        f"{cond}__matched__semantic_predicate_rate",
        f"{cond}_predicate_rate",
    )
    mismatched_semantic_keys = (
        f"{cond}__mismatched_source__semantic_predicate_rate",
        f"{cond}__mismatched_source_predicate_rate",
    )
    no_steer_keys = (
        f"{cond}__matched__no_steer_predicate_rate",
        f"{cond}__no_steer_predicate_rate",
    )

    def _first(keys):
        for k in keys:
            if k in summary:
                return summary[k]
        return None

    matched = _first(matched_semantic_keys)
    mismatched = _first(mismatched_semantic_keys)
    no_steer = _first(no_steer_keys)
    delta_cw = None
    if isinstance(matched, (int, float)) and isinstance(mismatched, (int, float)):
        delta_cw = matched - mismatched
    steer_lift = None
    if isinstance(matched, (int, float)) and isinstance(no_steer, (int, float)):
        steer_lift = matched - no_steer
    return {
        "alpha_scale": summary.get("alpha_scale"),
        "condition": cond,
        "matched_semantic": matched,
        "mismatched_semantic": mismatched,
        "matched_no_steer": no_steer,
        "delta_cw": delta_cw,
        "steer_lift": steer_lift,
        "n": summary.get("n"),
        "elapsed_s": summary.get("elapsed_s"),
        "error": summary.get("error"),
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    alphas = _parse_float_list(args.alpha_scales)
    if not alphas:
        print("FATAL: --alpha-scales is empty", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cond_list = [c.strip() for c in args.conditions.split(",") if c.strip()]
    sweep_summary: dict = {
        "version": 1,
        "started_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "alpha_scales": alphas,
        "conditions": cond_list,
        "intent_arms": args.intent_arms,
        "causal_arms": args.causal_arms,
        "sim_placement": args.sim_placement,
        "n_samples": args.n_samples,
        "policy_host": args.policy_host,
        "policy_port": args.policy_port,
        "compare_script": str(COMPARE_SCRIPT),
        "rows": [],
        "per_alpha_json": [],
    }

    for alpha in alphas:
        out_json = out_dir / f"alpha_{alpha:.3f}.json"
        summ = _run_one_alpha(args, alpha, out_json)
        sweep_summary["per_alpha_json"].append(str(out_json))
        for cond in cond_list:
            row = _extract_row(summ, cond)
            sweep_summary["rows"].append(row)
            if row.get("error"):
                print(f"  alpha={alpha:.3f} {cond}: ERROR {row['error']}")
            else:
                d = row.get("delta_cw")
                m = row.get("succ_matched_semantic")
                w = row.get("succ_mismatched_semantic")
                lift = row.get("steer_lift")
                d_str = f"{d:+.2%}" if isinstance(d, (int, float)) else "-"
                m_str = f"{m:.2%}" if isinstance(m, (int, float)) else "-"
                w_str = f"{w:.2%}" if isinstance(w, (int, float)) else "-"
                l_str = f"{lift:+.2%}" if isinstance(lift, (int, float)) else "-"
                print(
                    f"  alpha={alpha:.3f} {cond:>8}: matched={m_str} "
                    f"mismatched={w_str} Δ_cw={d_str} steer_lift={l_str}",
                    flush=True,
                )

    sweep_summary["ended_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(sweep_summary, indent=2))
    print(f"\nWrote sweep summary → {summary_path}")

    # Stage-0 decision rule (advisory, not enforced).
    valid_rows = [r for r in sweep_summary["rows"]
                  if r.get("delta_cw") is not None and r.get("error") is None]
    if valid_rows:
        best = max(valid_rows, key=lambda r: r["delta_cw"])
        worst = min(valid_rows, key=lambda r: r["delta_cw"])
        print(
            f"\nStage-0 decision: best Δ_cw = {best['delta_cw']:+.2%} "
            f"@ alpha={best['alpha_scale']:.3f} ({best['condition']}); "
            f"worst = {worst['delta_cw']:+.2%} @ alpha={worst['alpha_scale']:.3f}"
        )
        if best["delta_cw"] >= 0.05:
            print(
                "  → DOSE-MISCALIBRATION confirmed: some alpha lifts matched "
                "over mismatched by ≥5pp. Pin this alpha for downstream "
                "evals and skip the architectural escalation."
            )
        elif all(-0.02 <= r["delta_cw"] <= 0.02 for r in valid_rows):
            print(
                "  → CODEC FAILURE confirmed: Δ_cw stays in [-2pp, +2pp] "
                "across every alpha. Proceed to Stage 2 (image_patch-only "
                "retrain) and Stage 3 (spatial AR)."
            )
        else:
            print(
                "  → INCONCLUSIVE: some movement but no clear winner. "
                "Inspect per-alpha JSONs; consider widening the sample size "
                "or alpha range."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
