"""A/B decision script for the action-head consistency overnight run.

Reads both arms' ``metrics.jsonl`` and emits a single line to stdout:

  - ``PROMOTE_TO_MAIN`` when **all** of these are true:
      1. Arm B completed all target steps without crashing
         (>= ``--min-steps`` rows in metrics.jsonl).
      2. No NaN/Inf in ``loss``, ``ce``, ``ar_mse``, ``ar_nce``,
         ``action_consistency_loss`` across Arm B's rows.
      3. Arm B's ``action_consistency_loss`` at the last step is
         <= ``--consistency-growth-factor`` * its value at the early
         step (default 1.5x earliest non-zero diagnostic).
      4. Arm B's ``ce`` at the last step is within ``--ce-delta`` (+0.3)
         of Arm A's ``ce`` at the last step (AV not catastrophically
         destabilized).
  - ``FALLBACK_TO_V4_ONLY`` otherwise.

Designed for downstream bash:

  ``DECISION=$(python scripts/eval/decide_consistency_ab.py ...)``

A non-zero exit code means the script could not reach a decision (missing
files, malformed metrics, etc.). The caller should treat that as
``FALLBACK_TO_V4_ONLY``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _load_train_rows(metrics_path: Path) -> list[dict]:
    rows: list[dict] = []
    with metrics_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("phase") != "train":
                continue
            rows.append(r)
    return rows


def _finite(*vals: float) -> bool:
    for v in vals:
        try:
            if not math.isfinite(float(v)):
                return False
        except Exception:
            return False
    return True


def _last_step_value(rows: list[dict], key: str) -> float | None:
    for r in reversed(rows):
        if key in r and r[key] is not None:
            try:
                return float(r[key])
            except Exception:
                continue
    return None


def _first_nonzero(rows: list[dict], key: str) -> float | None:
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        try:
            x = float(v)
        except Exception:
            continue
        if x != 0.0:
            return x
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control-dir", required=True, help="Arm A output dir (lambda=0).")
    p.add_argument("--consistency-dir", required=True, help="Arm B output dir (lambda>0).")
    p.add_argument("--min-steps", type=int, default=300)
    p.add_argument("--consistency-growth-factor", type=float, default=1.5)
    p.add_argument("--ce-delta", type=float, default=0.3)
    p.add_argument(
        "--reasons-file",
        default=None,
        help="Optional path to write the verbose reasons; stdout is kept "
             "machine-friendly so callers can pipe into a bash branch.",
    )
    args = p.parse_args()

    reasons: list[str] = []

    control_metrics = Path(args.control_dir) / "metrics.jsonl"
    consist_metrics = Path(args.consistency_dir) / "metrics.jsonl"

    def fail(reason: str) -> "Never":
        reasons.append(reason)
        if args.reasons_file:
            Path(args.reasons_file).write_text("\n".join(reasons) + "\n")
        print("FALLBACK_TO_V4_ONLY", flush=True)
        sys.exit(0)

    if not control_metrics.exists():
        fail(f"missing control metrics: {control_metrics}")
    if not consist_metrics.exists():
        fail(f"missing consistency metrics: {consist_metrics}")

    control_rows = _load_train_rows(control_metrics)
    consist_rows = _load_train_rows(consist_metrics)

    reasons.append(
        f"control train_rows={len(control_rows)} "
        f"consistency train_rows={len(consist_rows)}"
    )

    # Rule 1: Arm B completed >= min_steps.
    if len(consist_rows) < args.min_steps // 5:
        # We log every 5 steps -> 400 steps gives ~80 train rows; require at
        # least min_steps//5 logged rows as a completion signal.
        fail(
            f"consistency arm produced only {len(consist_rows)} train rows; "
            f"need >= {args.min_steps // 5} (min_steps={args.min_steps}, "
            "log_every=5)."
        )

    # Rule 2: No NaN/Inf in the headline columns.
    bad_keys = ("loss", "ce", "ar_mse", "ar_nce", "action_consistency_loss")
    for r in consist_rows:
        present = [k for k in bad_keys if k in r and r[k] is not None]
        vals = [r[k] for k in present]
        if not _finite(*vals):
            fail(f"NaN/Inf in consistency metrics at step {r.get('step')}: {dict(zip(present, vals))}")

    # Rule 3: action_consistency_loss not growing pathologically.
    first_acl = _first_nonzero(consist_rows, "action_consistency_loss")
    last_acl = _last_step_value(consist_rows, "action_consistency_loss")
    if first_acl is None:
        fail("consistency arm never fired action_consistency_loss > 0; check cadence/manifest")
    if last_acl is None:
        fail("consistency arm missing final action_consistency_loss")
    growth = last_acl / max(abs(first_acl), 1e-9)
    reasons.append(
        f"action_consistency_loss first={first_acl:.4f} "
        f"last={last_acl:.4f} growth={growth:.2f}x "
        f"limit={args.consistency_growth_factor:.2f}x"
    )
    if growth > args.consistency_growth_factor:
        fail(
            f"action_consistency_loss grew {growth:.2f}x "
            f"(> {args.consistency_growth_factor:.2f}x); regressing not training"
        )

    # Rule 4: AV CE not catastrophically destabilized vs control.
    ce_control = _last_step_value(control_rows, "ce")
    ce_consist = _last_step_value(consist_rows, "ce")
    if ce_control is None or ce_consist is None:
        fail(
            f"missing final ce: control={ce_control} consistency={ce_consist}"
        )
    reasons.append(
        f"final_ce control={ce_control:.4f} consistency={ce_consist:.4f} "
        f"delta={ce_consist - ce_control:+.4f} budget=+{args.ce_delta:.2f}"
    )
    if (ce_consist - ce_control) > args.ce_delta:
        fail(
            f"consistency arm CE regressed by {ce_consist - ce_control:+.4f} "
            f"(> +{args.ce_delta:.4f}); AV destabilized"
        )

    reasons.append("PROMOTE_TO_MAIN")
    if args.reasons_file:
        Path(args.reasons_file).write_text("\n".join(reasons) + "\n")
    print("PROMOTE_TO_MAIN", flush=True)


if __name__ == "__main__":
    main()
