#!/usr/bin/env python3
"""Export a JSON snapshot for the technical writeup website.

Reads the artifacts already produced by the eval pipeline:

  * V3 scorecard JSON       (build_v3_scorecard.py)
  * Steerability metrics    (steerability_eval.py -> metrics.json)
  * AV fidelity judge means (steerability_eval.py -> av_metrics.json)
  * (optional) SFT metrics  (run_sft.py -> metrics.jsonl, val rows)

Writes:

  * website/src/data/snapshot.json   (committed, drives all charts)
  * website/public/figures/<png>     (copied from steerability run)

The script is idempotent and side-effect-only inside the website tree.
Run it whenever evals are refreshed::

    python scripts/website/export_site_data.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_SCORECARD = REPO / "data" / "sft" / "libero_4suite_v3" / "v3_scorecard.json"
DEFAULT_TRAIN_METRICS = REPO / "data" / "sft" / "libero_4suite_v3" / "metrics.jsonl"
DEFAULT_STEER_DIR = REPO / "data" / "eval" / "steerability_v1_vs_v3"
DEFAULT_OUT = REPO / "website" / "src" / "data" / "snapshot.json"
DEFAULT_FIGS_OUT = REPO / "website" / "public" / "figures"

_FIG_NAMES = (
    "per_object_displacement.png",
    "per_object_min_ee_distance.png",
)


def _load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _flatten_judge(av_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert nested av_metrics.json into a flat list of judge rows."""
    rows: list[dict[str, Any]] = []
    for key, blob in av_metrics.items():
        # key is e.g. "libero_4suite_v3__ar@libero_4suite_holdout"
        ckpt, _, holdout = key.partition("@")
        ckpt = ckpt.replace("__ar", "").rstrip("_")
        per_variant = blob.get("per_variant_mean", {})
        n_rows = blob.get("n_rows")
        for variant in ("gold", "av_pred"):
            v = per_variant.get(variant)
            if not v:
                continue
            rows.append(
                {
                    "checkpoint": ckpt,
                    "holdout": holdout,
                    "source": variant,
                    "grounding": v.get("grounding_pass_rate"),
                    "appropriateness": v.get("appropriateness_pass_rate"),
                    "template_distinguishable": v.get(
                        "template_distinguishable_pass_rate"
                    ),
                    "n_rows": n_rows,
                }
            )
    return rows


def _flatten_sim(steer_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Reduce steerability metrics.json to one row per condition."""
    rows: list[dict[str, Any]] = []
    for cond, blob in steer_metrics.get("conditions", {}).items():
        overall = blob.get("overall", {})
        rows.append(
            {
                "condition": cond,
                "success_any": overall.get("success_any_rate"),
                "success_final": overall.get("success_final_rate"),
                "mean_steps": overall.get("mean_n_steps"),
                "target_disp_m": overall.get("mean_target_displacement"),
                "target_min_ee_m": overall.get("mean_target_min_ee_distance"),
                "target_winner_rate": overall.get("target_winner_rate"),
                # Bowl is the manipulated object on the only env we ran.
                "bowl_disp_m": overall.get("displacement", {}).get(
                    "akita_black_bowl_1_main"
                ),
            }
        )
    return rows


def _training_curve(
    metrics_path: Path, max_points: int = 40
) -> list[dict[str, Any]]:
    """Pick val rows from SFT metrics.jsonl, keep step/fve/closed_greedy_fve."""
    if not metrics_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with metrics_path.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if r.get("phase") not in {"val", "final"}:
                continue
            rows.append(
                {
                    "step": r.get("step"),
                    "fve": r.get("fve"),
                    "closed_greedy_fve": r.get("closed_greedy/fve"),
                }
            )
    # Subsample evenly if too long.
    if len(rows) > max_points:
        idxs = [round(i * (len(rows) - 1) / (max_points - 1)) for i in range(max_points)]
        rows = [rows[i] for i in idxs]
    return rows


def _copy_figures(steer_dir: Path, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in _FIG_NAMES:
        src = steer_dir / "figures" / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
            copied.append(name)
    return copied


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    p.add_argument("--steer-dir", type=Path, default=DEFAULT_STEER_DIR)
    p.add_argument("--train-metrics", type=Path, default=DEFAULT_TRAIN_METRICS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--figures-out", type=Path, default=DEFAULT_FIGS_OUT)
    args = p.parse_args(argv)

    if not args.scorecard.exists():
        print(f"[error] scorecard not found: {args.scorecard}", file=sys.stderr)
        return 2
    scorecard = _load_json(args.scorecard)

    av_metrics_path = args.steer_dir / "av_metrics.json"
    sim_metrics_path = args.steer_dir / "metrics.json"
    judge: list[dict[str, Any]] = []
    sim: list[dict[str, Any]] = []
    if av_metrics_path.exists():
        judge = _flatten_judge(_load_json(av_metrics_path))
    else:
        print(f"[warn] missing {av_metrics_path}; judge chart will be empty",
              file=sys.stderr)
    if sim_metrics_path.exists():
        sim = _flatten_sim(_load_json(sim_metrics_path))
    else:
        print(f"[warn] missing {sim_metrics_path}; sim chart will be empty",
              file=sys.stderr)

    training = _training_curve(args.train_metrics)

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scorecard": scorecard,
        "judge": judge,
        "sim": sim,
        "training": training,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(snapshot, indent=2) + "\n")
    print(f"[ok] wrote {args.out}  ({args.out.stat().st_size} bytes)")

    figs = _copy_figures(args.steer_dir, args.figures_out)
    if figs:
        print(f"[ok] copied {len(figs)} figure(s) to {args.figures_out}: "
              f"{', '.join(figs)}")
    else:
        print(f"[warn] no figures copied from {args.steer_dir}/figures",
              file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
