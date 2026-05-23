#!/usr/bin/env python
"""Build a publishable GRPO sim-steer scorecard.

Reads a held-out CF compare JSON (``compare_cf_steer_checkpoints.py``) and an
optional GRPO ``metrics.jsonl`` and emits a single ``grpo_steer_scorecard.json``
plus a human-readable console table.

The headline metric is **predicate rate on the steered ``target_task``** under
the matched/semantic arm — explicitly labeled as an **xyz-heuristic**, not
LIBERO native BDDL success. BDDL success is reported as a labeled secondary
column so reviewers can see both side-by-side.

Bands are deliberately conservative for a *publishable* CoRL claim:

  | Metric                                            | Pass    | Warn       | Fail   |
  | ------------------------------------------------- | ------- | ---------- | ------ |
  | delta_predicate_rate_grpo_minus_sft               | >= +10pp| 0-10pp     | <0     |
  | semantic_gap_predicate (grpo_av)                  | >= +10pp| 0-10pp     | <0     |
  | steer_lift_predicate (grpo_av)                    | >= +10pp| 0-10pp     | <0     |
  | causal_specificity_predicate (grpo_av)            | >= +10pp| 0-10pp     | <0     |
  | placement_specificity_predicate (grpo_av)         | >= +5pp | 0-5pp      | <0     |
  | grpo_av_predicate_rate (absolute)                 | >= 25%  | 10-25%     | <10%   |
  | closed_greedy_cosine (val guardrail)              | >= 0.64 | 0.55-0.64  | <0.55  |

The "publishable" verdict (a CoRL-positive result on steering) requires:
  (a) delta_predicate_rate_grpo_minus_sft PASSES, AND
  (b) semantic_gap_predicate (grpo_av) PASSES (if mismatched_source arm ran),
       OR the audit-style narrative explicitly notes "no semantic gap", AND
  (c) closed_greedy_cosine stays >= warn (no AV collapse).

Under eval-v2 (``eval_protocol=language_swap``, the new default), the
``semantic_gap_predicate`` becomes the primary success signal: the
matched / mismatched_source arms feed the policy distinct intent texts
on the same target scene, so a non-zero gap can only come from the AV
caption itself moving the policy. ``steer_lift_predicate`` (semantic -
no_steer) audits whether steering helps at all over an unsteered
baseline. Legacy compare JSONs (``eval_protocol=legacy``) are still
read for backward compat but the gap there is structurally near zero
and should not be cited as evidence of language steering.

A run can still be a publishable **audit / negative result** if (a)/(b) FAIL
but the script is rerun with ``--narrative audit`` — that flips the verdict
"AUDIT_PASS" without changing the underlying numbers.

Usage::

    PYTHONPATH=src python scripts/eval/build_grpo_steer_scorecard.py \\
        --compare-json     data/eval/cf_steer_sft_vs_grpo_v2.json \\
        --grpo-metrics     data/grpo/.../metrics.jsonl \\
        --out-json         data/eval/grpo_steer_scorecard.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class Band:
    pass_at: float
    warn_at: float
    higher_is_better: bool = True

    def evaluate(self, value: float | None) -> str:
        if value is None:
            return "NA"
        if self.higher_is_better:
            if value >= self.pass_at:
                return "PASS"
            if value >= self.warn_at:
                return "WARN"
            return "FAIL"
        if value <= self.pass_at:
            return "PASS"
        if value <= self.warn_at:
            return "WARN"
        return "FAIL"


BANDS: dict[str, Band] = {
    "delta_predicate_rate_grpo_minus_sft": Band(0.10, 0.00),
    "grpo_semantic_gap_predicate":         Band(0.10, 0.00),
    "grpo_steer_lift_predicate":           Band(0.10, 0.00),
    "grpo_causal_specificity_predicate":   Band(0.10, 0.00),
    "grpo_placement_specificity_predicate": Band(0.05, 0.00),
    "grpo_av_predicate_rate":              Band(0.25, 0.10),
    "closed_greedy_cosine":                Band(0.64, 0.55),
}

REQUIRED_FOR_PUBLISHABLE = (
    "delta_predicate_rate_grpo_minus_sft",
    "grpo_semantic_gap_predicate",
)


def _read_compare(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    cfg = obj.get("config", {}) or {}
    intent_arms = obj.get("intent_arms") or ["matched"]
    causal_arms = obj.get("causal_arms") or ["semantic"]
    out: dict[str, Any] = {
        "compare_json": str(path),
        "n_samples": obj.get("n"),
        "intent_arms": intent_arms,
        "causal_arms": causal_arms,
        "eval_protocol": obj.get("eval_protocol")
            or cfg.get("eval_protocol")
            or "legacy",
        "config": cfg,
        # Headline metrics from compare summary (already labeled correctly).
        "sft_av_predicate_rate":  obj.get("sft_av_predicate_rate"),
        "grpo_av_predicate_rate": obj.get("grpo_av_predicate_rate"),
        "sft_av_success_bddl_native_rate": obj.get(
            "sft_av_success_any_rate"
        ),
        "grpo_av_success_bddl_native_rate": obj.get(
            "grpo_av_success_any_rate"
        ),
        "delta_predicate_rate_grpo_minus_sft": obj.get(
            "delta_predicate_rate_grpo_minus_sft"
        ),
        "delta_success_bddl_native_rate_grpo_minus_sft": obj.get(
            "delta_success_bddl_native_rate_grpo_minus_sft"
        ),
        "paired_wins_grpo_predicate": obj.get("paired_wins_grpo_predicate"),
        "paired_losses_grpo_predicate": obj.get("paired_losses_grpo_predicate"),
        # Semantic / causal specificity (None if those arms didn't run).
        "sft_semantic_gap_predicate":  obj.get("sft_av_semantic_gap_predicate"),
        "grpo_semantic_gap_predicate": obj.get("grpo_av_semantic_gap_predicate"),
        "sft_causal_specificity_predicate": obj.get(
            "sft_av_causal_specificity_predicate"
        ),
        "grpo_causal_specificity_predicate": obj.get(
            "grpo_av_causal_specificity_predicate"
        ),
        "sft_placement_specificity_predicate": obj.get(
            "sft_av_placement_specificity_predicate"
        ),
        "grpo_placement_specificity_predicate": obj.get(
            "grpo_av_placement_specificity_predicate"
        ),
        "sft_steer_lift_predicate": obj.get(
            "sft_av_steer_lift_predicate"
        ),
        "grpo_steer_lift_predicate": obj.get(
            "grpo_av_steer_lift_predicate"
        ),
    }

    # Per-target_task breakdown (predicate rate, both checkpoints).
    samples = obj.get("samples") or []
    by_task: dict[str, dict[str, list[float]]] = {}
    for s in samples:
        task = s.get("target_task") or "(unknown)"
        for cn in ("sft_av", "grpo_av"):
            c = (s.get("conditions") or {}).get(cn)
            if not c or c.get("error") is not None or "skipped_reason" in c:
                continue
            by_task.setdefault(task, {}).setdefault(cn, []).append(
                float(c.get("predicate", 0.0))
            )
    per_task: dict[str, dict[str, float | int]] = {}
    for task, by_cn in sorted(by_task.items()):
        row: dict[str, float | int] = {}
        for cn, vals in by_cn.items():
            row[f"{cn}_predicate_rate"] = (
                sum(1 for v in vals if v > 0) / len(vals)
            )
            row[f"{cn}_n"] = len(vals)
        if "sft_av_predicate_rate" in row and "grpo_av_predicate_rate" in row:
            row["delta_predicate_rate_grpo_minus_sft"] = (
                row["grpo_av_predicate_rate"] - row["sft_av_predicate_rate"]
            )
        per_task[task] = row
    out["per_target_task"] = per_task

    # Per-slice breakdown (matched_bddl vs cross_scene_cf), inferred from the
    # scoring block stamped on each sample by compare_cf_steer_checkpoints.py.
    by_slice: dict[str, dict[str, list[float]]] = {}
    for s in samples:
        sc = s.get("scoring") or {}
        env_match = sc.get("env_matches_scored_task")
        if env_match is True:
            slice_name = "env_matched"
        elif env_match is False:
            slice_name = "cross_scene_cf"
        else:
            slice_name = "unknown"
        for cn in ("sft_av", "grpo_av"):
            c = (s.get("conditions") or {}).get(cn)
            if not c or c.get("error") is not None or "skipped_reason" in c:
                continue
            by_slice.setdefault(slice_name, {}).setdefault(cn, []).append(
                float(c.get("predicate", 0.0))
            )
    per_slice: dict[str, dict[str, float | int]] = {}
    for slice_name, by_cn in sorted(by_slice.items()):
        row = {}
        for cn, vals in by_cn.items():
            row[f"{cn}_predicate_rate"] = (
                sum(1 for v in vals if v > 0) / len(vals)
            )
            row[f"{cn}_n"] = len(vals)
        per_slice[slice_name] = row
    out["per_slice"] = per_slice
    return out


def _read_grpo_metrics(path: Path | None) -> dict[str, float | None]:
    """Pull the latest val-side ``closed_greedy/cosine`` from metrics.jsonl."""
    if path is None or not path.exists():
        return {"closed_greedy_cosine": None}
    latest: float | None = None
    latest_step: int | None = None
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            val = obj.get("closed_greedy/cosine") or obj.get("closed_greedy_cosine")
            if val is None:
                continue
            step = obj.get("step")
            if latest is None or (step is not None and (latest_step or -1) < step):
                latest = float(val)
                latest_step = step
    return {"closed_greedy_cosine": latest, "closed_greedy_cosine_step": latest_step}


def _band_value(name: str, value: float | None) -> str:
    band = BANDS.get(name)
    if band is None:
        return "NA"
    return band.evaluate(value)


def _print_table(rows: list[tuple[str, float | None, str, str]]) -> None:
    print(f"{'metric':<48}{'value':>10}{'band':>8}  note")
    print("-" * 90)
    for name, value, band, note in rows:
        v_str = "    n/a" if value is None else f"{value * 100:+7.2f}%"
        print(f"{name:<48}{v_str:>10}{band:>8}  {note}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--compare-json", required=True,
                   help="Output of compare_cf_steer_checkpoints.py")
    p.add_argument("--grpo-metrics", default=None,
                   help="GRPO metrics.jsonl for closed_greedy/cosine guardrail.")
    p.add_argument("--out-json", required=True,
                   help="Path to write the scorecard JSON.")
    p.add_argument("--narrative", choices=("publishable", "audit"),
                   default="publishable",
                   help="Verdict mode: 'publishable' requires beat-SFT + "
                        "semantic gap; 'audit' treats negative results as a "
                        "valid contribution (always PASS).")
    args = p.parse_args(argv)

    compare = _read_compare(Path(args.compare_json))
    grpo_metrics = _read_grpo_metrics(
        Path(args.grpo_metrics) if args.grpo_metrics else None
    )

    # Headline rows for console + JSON.
    metrics: dict[str, float | None] = {
        "delta_predicate_rate_grpo_minus_sft":
            compare.get("delta_predicate_rate_grpo_minus_sft"),
        "grpo_av_predicate_rate":
            compare.get("grpo_av_predicate_rate"),
        "sft_av_predicate_rate":
            compare.get("sft_av_predicate_rate"),
        "grpo_av_success_bddl_native_rate":
            compare.get("grpo_av_success_bddl_native_rate"),
        "sft_av_success_bddl_native_rate":
            compare.get("sft_av_success_bddl_native_rate"),
        "grpo_semantic_gap_predicate":
            compare.get("grpo_semantic_gap_predicate"),
        "sft_semantic_gap_predicate":
            compare.get("sft_semantic_gap_predicate"),
        "grpo_steer_lift_predicate":
            compare.get("grpo_steer_lift_predicate"),
        "sft_steer_lift_predicate":
            compare.get("sft_steer_lift_predicate"),
        "grpo_causal_specificity_predicate":
            compare.get("grpo_causal_specificity_predicate"),
        "grpo_placement_specificity_predicate":
            compare.get("grpo_placement_specificity_predicate"),
        "closed_greedy_cosine":
            grpo_metrics.get("closed_greedy_cosine"),
    }

    bands = {name: _band_value(name, val) for name, val in metrics.items()}

    # Overall verdict.
    if args.narrative == "audit":
        verdict = "AUDIT_PASS"
        verdict_note = (
            "narrative=audit: numeric outcome treated as a valid (audit) "
            "contribution regardless of beat-SFT band."
        )
    else:
        required_states = [bands.get(k) for k in REQUIRED_FOR_PUBLISHABLE]
        if any(s == "FAIL" for s in required_states):
            verdict = "FAIL"
        elif any(s == "WARN" for s in required_states):
            verdict = "WARN"
        elif all(s == "PASS" for s in required_states):
            verdict = "PASS"
        else:
            # Mostly NA — at minimum the delta must be present.
            verdict = "NA"
        cos_band = bands.get("closed_greedy_cosine", "NA")
        if verdict == "PASS" and cos_band == "FAIL":
            verdict = "WARN"
        verdict_note = (
            f"narrative=publishable: requires "
            f"{', '.join(REQUIRED_FOR_PUBLISHABLE)} all PASS"
        )

    out_obj: dict[str, Any] = {
        "schema_version": 1,
        "verdict": verdict,
        "verdict_note": verdict_note,
        "narrative": args.narrative,
        "metrics": metrics,
        "bands": bands,
        "compare": compare,
        "grpo_metrics": grpo_metrics,
        "headline_metric_definitions": {
            "predicate_rate":
                "Fraction of held-out CF rollouts where the steered "
                "target_task xyz heuristic fires at least once. NOT LIBERO "
                "BDDL native success.",
            "success_bddl_native_rate":
                "Fraction of rollouts where LIBERO's loaded-scene BDDL "
                "success fires. Often != steered intent on cross-scene CF.",
            "semantic_gap_predicate":
                "matched_intent predicate rate - mismatched_source_intent "
                "predicate rate; positive value means language is causal "
                "for steering. Under eval_protocol=language_swap this is "
                "the primary publishable success signal.",
            "steer_lift_predicate":
                "matched/semantic predicate rate - matched/no_steer "
                "predicate rate; positive means the AR-injected steer "
                "adds reward over the unsteered base policy.",
            "causal_specificity_predicate":
                "semantic predicate rate - matched_null predicate rate; "
                "positive means the AR vector beats norm-matched noise.",
            "placement_specificity_predicate":
                "semantic predicate rate - wrong_placement predicate rate; "
                "positive means token-role placement matters.",
        },
        "bands_definition": {k: asdict(b) for k, b in BANDS.items()},
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, indent=2))

    rows = [
        ("delta_predicate_rate_grpo_minus_sft",
         metrics["delta_predicate_rate_grpo_minus_sft"],
         bands["delta_predicate_rate_grpo_minus_sft"],
         "GRPO - SFT, matched/semantic"),
        ("grpo_av_predicate_rate",
         metrics["grpo_av_predicate_rate"],
         bands["grpo_av_predicate_rate"],
         "absolute"),
        ("sft_av_predicate_rate",
         metrics["sft_av_predicate_rate"], "REF", "absolute (baseline)"),
        ("grpo_av_success_bddl_native_rate",
         metrics["grpo_av_success_bddl_native_rate"], "REF",
         "loaded scene, secondary"),
        ("grpo_semantic_gap_predicate",
         metrics["grpo_semantic_gap_predicate"],
         bands["grpo_semantic_gap_predicate"],
         "matched - mismatched_source"),
        ("grpo_steer_lift_predicate",
         metrics["grpo_steer_lift_predicate"],
         bands["grpo_steer_lift_predicate"],
         "semantic - no_steer"),
        ("grpo_causal_specificity_predicate",
         metrics["grpo_causal_specificity_predicate"],
         bands["grpo_causal_specificity_predicate"],
         "semantic - matched_null"),
        ("grpo_placement_specificity_predicate",
         metrics["grpo_placement_specificity_predicate"],
         bands["grpo_placement_specificity_predicate"],
         "semantic - wrong_placement"),
        ("closed_greedy_cosine",
         metrics["closed_greedy_cosine"],
         bands["closed_greedy_cosine"],
         "AV recon guardrail (NOT a steer metric)"),
    ]
    protocol_note = compare.get("eval_protocol") or "legacy"
    print(
        f"GRPO steer scorecard (n={compare.get('n_samples')}, "
        f"eval_protocol={protocol_note}):"
    )
    _print_table(rows)
    print("-" * 90)
    print(f"  Verdict: {verdict}  ({verdict_note})")
    print(f"  Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
