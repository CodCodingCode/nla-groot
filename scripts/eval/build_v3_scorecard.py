#!/usr/bin/env python
"""Build the unified V3 LIBERO scorecard.

Reads every eval artifact produced for an SFT checkpoint and emits a single
``v3_scorecard.json`` plus a human-readable console summary that says
"is V3 real" in one glance.

Inputs (paths are auto-discovered relative to ``--ckpt-dir`` but each may be
overridden):

  - ``<ckpt>/retrieval_margin.json``   (from scripts/eval/closed_loop_retrieval.py)
  - ``<ckpt>/llm_judge.jsonl``          (from scripts/eval/llm_judge_av_captions.py)
  - ``<ckpt>/metrics.jsonl``            (training-time val metrics)
  - ``<ckpt>/sim_ab.json``              (P2 future: scripts/eval/closed_loop_sim_ab.py)

Bands (V3 LIBERO Eval Refactor plan):

  | Metric                              | Pass    | Warn       | Fail   |
  | ----------------------------------- | ------- | ---------- | ------ |
  | retrieval_margin                    | >= 0.05 | 0.02-0.05  | <0.02  |
  | retrieval@1                         | >= 25%  | 15-25%     | <15%   |
  | judge_grounding_specific_pct        | >= 55%  | 40-55%     | <40%   |
  | judge_anti_template_specific_pct    | >= 50%  | 30-50%     | <30%   |
  | sim_correct_minus_wrong (P2)        | >= +5pp | 0-5pp      | <=0    |
  | sim_correct_success (P2)            | >= 30%  | 15-30%     | <15%   |
  | closed_greedy_cosine (training)     | >= 0.55 | 0.40-0.55  | <0.40  |
                                                                    (reference)

A V3 run **PASSES overall** iff:
  (a) retrieval_margin                    PASSES, AND
  (b) judge_grounding_specific_pct        PASSES, AND
  (c) sim_correct_minus_wrong             PASSES (if P2 data is present;
                                            otherwise required-PASS gate is
                                            relaxed to "judge_anti_template
                                            PASSES").
Otherwise WARN if any band is WARN, else FAIL.

Usage::

    PYTHONPATH=src python scripts/eval/build_v3_scorecard.py \\
        --ckpt-dir   data/sft/libero_4suite_v3 \\
        --out-json   data/sft/libero_4suite_v3/v3_scorecard.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Band logic
# ---------------------------------------------------------------------------

@dataclass
class Band:
    """Pass/Warn/Fail thresholds.

    ``higher_is_better`` selects the comparison direction. For example with
    ``pass_at=0.05, warn_at=0.02, higher_is_better=True``:
      value >= 0.05  -> PASS
      0.02 <= v <  0.05 -> WARN
      value <  0.02  -> FAIL
    """
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
        # lower-is-better path (none used today; reserved for losses)
        if value <= self.pass_at:
            return "PASS"
        if value <= self.warn_at:
            return "WARN"
        return "FAIL"


# Bands per the V3 LIBERO Eval Refactor plan. Keep these as plain literals
# rather than overridable CLI args -- they're contract, not config.
BANDS: dict[str, Band] = {
    "retrieval_margin":                 Band(pass_at=0.05, warn_at=0.02),
    "retrieval_at_1":                   Band(pass_at=0.25, warn_at=0.15),
    "retrieval_at_5":                   Band(pass_at=0.55, warn_at=0.40),
    "judge_grounding_specific_pct":     Band(pass_at=0.55, warn_at=0.40),
    "judge_appropriateness_pct":        Band(pass_at=0.80, warn_at=0.60),
    "judge_anti_template_specific_pct": Band(pass_at=0.50, warn_at=0.30),
    "closed_greedy_cosine":             Band(pass_at=0.55, warn_at=0.40),
    # P2 sim-A/B bands (extended in p2-scorecard-extend; values here so a
    # future sim_ab.json drop-in works without code edits).
    "sim_correct_minus_wrong":          Band(pass_at=0.05, warn_at=0.00),
    "sim_correct_success":              Band(pass_at=0.30, warn_at=0.15),
    "sim_correct_minus_baseline_floor": Band(
        pass_at=-0.10, warn_at=-0.30, higher_is_better=True,
    ),
    # Negative-control informational signal: how much *wrong*-captioned
    # steering depresses success vs no-steer baseline. We want this to be
    # close to zero or slightly negative (steering hurts the wrong way),
    # but a large positive value means the wrong arm is *helping*, which
    # would be a smoking gun for non-grounding.
    "sim_wrong_minus_baseline":         Band(
        pass_at=0.00, warn_at=0.05, higher_is_better=False,
    ),
}

# The subset of bands that must all PASS to call the whole run a PASS.
# ``sim_correct_minus_wrong`` is added only if P2 data is present, otherwise
# we substitute the judge anti-template axis as the gate.
REQUIRED_FOR_PASS_NO_SIM = (
    "retrieval_margin",
    "judge_grounding_specific_pct",
    "judge_anti_template_specific_pct",
)
REQUIRED_FOR_PASS_WITH_SIM = (
    "retrieval_margin",
    "judge_grounding_specific_pct",
    "sim_correct_minus_wrong",
)

# Metrics that affect the overall verdict (warn-aware) but are not in the
# required-pass set above. Reference-only signals live here.
INFORMATIONAL = (
    "retrieval_at_1",
    "retrieval_at_5",
    "judge_appropriateness_pct",
    "closed_greedy_cosine",
    "sim_correct_success",
    "sim_correct_minus_baseline_floor",
    "sim_wrong_minus_baseline",
)


# ---------------------------------------------------------------------------
# Source readers
# ---------------------------------------------------------------------------

def _read_retrieval(path: Path) -> dict[str, float | None]:
    """Return {retrieval_margin, retrieval_at_1, retrieval_at_5} from
    closed_loop_retrieval.py output, or all-Nones if file is missing."""
    if not path.exists():
        return {"retrieval_margin": None, "retrieval_at_1": None, "retrieval_at_5": None}
    obj = json.loads(path.read_text())
    return {
        "retrieval_margin": obj.get("margin"),
        "retrieval_at_1": obj.get("retrieval_at_1"),
        "retrieval_at_5": obj.get("retrieval_at_5"),
    }


def _read_judge(path: Path, variant: str = "av_pred") -> dict[str, float | None]:
    """Aggregate llm_judge_av_captions.py jsonl over rows where
    ``variant_id == variant`` (default: ``av_pred``)."""
    if not path.exists():
        return {
            "judge_grounding_specific_pct": None,
            "judge_appropriateness_pct": None,
            "judge_anti_template_specific_pct": None,
            "judge_n": 0,
            "judge_n_template": 0,
        }
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("variant_id") == variant and not obj.get("error"):
                rows.append(obj)
    n = len(rows)
    if n == 0:
        return {
            "judge_grounding_specific_pct": None,
            "judge_appropriateness_pct": None,
            "judge_anti_template_specific_pct": None,
            "judge_n": 0,
            "judge_n_template": 0,
        }
    b_pass = sum(
        1 for r in rows
        if (r.get("grounding") or {}).get("verdict") == "specific"
    )
    c_pass = sum(
        1 for r in rows
        if (r.get("appropriateness") or {}).get("verdict") == "appropriate"
    )
    d_rows = [r for r in rows if r.get("template_distinguishable")]
    n_d = len(d_rows)
    d_pass = sum(
        1 for r in d_rows
        if (r.get("template_distinguishable") or {}).get("verdict") == "specific"
    )
    return {
        "judge_grounding_specific_pct": b_pass / n,
        "judge_appropriateness_pct": c_pass / n,
        "judge_anti_template_specific_pct": (d_pass / n_d) if n_d else None,
        "judge_n": n,
        "judge_n_template": n_d,
    }


def _read_training_metrics(path: Path) -> dict[str, float | None]:
    """Pull the *last* val row from metrics.jsonl and surface its closed-loop
    cosine. Returns None if there are no val rows."""
    if not path.exists():
        return {"closed_greedy_cosine": None, "training_step": None, "training_phase": None}
    last_val: dict | None = None
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("phase") == "val":
                last_val = row
    if last_val is None:
        return {"closed_greedy_cosine": None, "training_step": None, "training_phase": None}
    cl = last_val.get("closed_greedy/cosine")
    return {
        "closed_greedy_cosine": cl,
        "training_step": last_val.get("step"),
        "training_phase": last_val.get("phase"),
    }


def _read_sim_ab(path: Path) -> dict[str, Any]:
    """Read closed_loop_sim_ab.py output and roll it up.

    Produces top-level scalar means (across the per-task arms) plus a
    per-suite breakdown that the scorecard surfaces in its config section.
    Missing file -> all-None / not-present.
    """
    out: dict[str, Any] = {
        "sim_correct_success": None,
        "sim_wrong_success": None,
        "sim_baseline_success": None,
        "sim_correct_minus_wrong": None,
        "sim_correct_minus_baseline_floor": None,
        "sim_wrong_minus_baseline": None,
        "sim_n_episodes": 0,
        "sim_per_suite": {},
        "sim_present": False,
    }
    if not path.exists():
        return out
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return out
    out["sim_present"] = True

    # Prefer pre-computed top-level means; fall back to recomputing from arms.
    out["sim_correct_success"] = obj.get("correct_success_mean")
    out["sim_wrong_success"] = obj.get("wrong_success_mean")
    out["sim_baseline_success"] = obj.get("baseline_success_mean")
    out["sim_n_episodes"] = obj.get("n_episodes_per_arm", 0)

    arms = obj.get("arms") or []
    # Per-suite breakdown: {suite: {arm: success_rate}}
    per_suite: dict[str, dict[str, float]] = {}
    for arm in arms:
        suite = arm.get("suite", "?")
        a = arm.get("arm", "?")
        sr = arm.get("success_rate")
        if isinstance(sr, (int, float)):
            per_suite.setdefault(suite, {})[a] = float(sr)
    out["sim_per_suite"] = per_suite

    # If top-level means missing, compute from arms.
    def _mean(arm_name: str) -> float | None:
        xs = [a.get("success_rate") for a in arms
              if a.get("arm") == arm_name
              and isinstance(a.get("success_rate"), (int, float))]
        return float(sum(xs) / len(xs)) if xs else None

    if out["sim_correct_success"] is None:
        out["sim_correct_success"] = _mean("correct")
    if out["sim_wrong_success"] is None:
        out["sim_wrong_success"] = _mean("wrong")
    if out["sim_baseline_success"] is None:
        out["sim_baseline_success"] = _mean("baseline")

    cs = out["sim_correct_success"]
    ws = out["sim_wrong_success"]
    bs = out["sim_baseline_success"]
    if cs is not None and ws is not None:
        out["sim_correct_minus_wrong"] = float(cs) - float(ws)
    if cs is not None and bs is not None:
        # We measure the *floor*: how far below baseline did steering push
        # success? Plan band: ``sim_correct_success - sim_baseline_success >= -0.10``.
        out["sim_correct_minus_baseline_floor"] = float(cs) - float(bs)
    if ws is not None and bs is not None:
        # Informational: wrong-arm vs baseline. We expect wrong steering to
        # *hurt* if the model is properly grounded; a small or positive
        # gap means wrong injection isn't actually moving behavior.
        out["sim_wrong_minus_baseline"] = float(ws) - float(bs)
    return out


# ---------------------------------------------------------------------------
# Verdict assembly
# ---------------------------------------------------------------------------

@dataclass
class MetricRow:
    name: str
    value: float | None
    threshold_pass: float
    threshold_warn: float
    higher_is_better: bool
    verdict: str         # PASS | WARN | FAIL | NA
    required_for_overall: bool


def _build_metric_rows(values: dict[str, Any]) -> list[MetricRow]:
    rows: list[MetricRow] = []
    has_sim = bool(values.get("sim_present"))
    required = (
        REQUIRED_FOR_PASS_WITH_SIM if has_sim else REQUIRED_FOR_PASS_NO_SIM
    )
    ordered_metrics = [m for m in required if m in BANDS]
    ordered_metrics += [m for m in INFORMATIONAL if m in BANDS and m not in required]
    # Also include sim_correct_minus_wrong if sim NOT present, just to surface
    # the gap, but mark it NA / not-required.
    if not has_sim and "sim_correct_minus_wrong" not in ordered_metrics:
        ordered_metrics.append("sim_correct_minus_wrong")
    for name in ordered_metrics:
        band = BANDS[name]
        v = values.get(name)
        verdict = band.evaluate(v if isinstance(v, (int, float)) else None)
        rows.append(MetricRow(
            name=name,
            value=v if isinstance(v, (int, float)) else None,
            threshold_pass=band.pass_at,
            threshold_warn=band.warn_at,
            higher_is_better=band.higher_is_better,
            verdict=verdict,
            required_for_overall=name in required,
        ))
    return rows


def _overall_verdict(rows: list[MetricRow]) -> str:
    """PASS iff all required rows PASS. FAIL iff any required row FAILs.
    WARN otherwise (including required NA / WARN, or any informational FAIL)."""
    required = [r for r in rows if r.required_for_overall]
    if any(r.verdict == "FAIL" for r in required):
        return "FAIL"
    if any(r.verdict in ("WARN", "NA") for r in required):
        return "WARN"
    # All required PASS. Informational FAIL still warrants a WARN at the
    # top-level so it doesn't get lost.
    info = [r for r in rows if not r.required_for_overall]
    if any(r.verdict == "FAIL" for r in info):
        return "WARN"
    return "PASS"


@dataclass
class Scorecard:
    checkpoint: str
    overall: str
    metrics: list[dict] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)
    config: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt-dir", required=True,
                   help="SFT run dir. We auto-discover the eval outputs under "
                        "it and write the scorecard there too.")
    p.add_argument("--retrieval-json", default=None,
                   help="Override path to retrieval_margin.json.")
    p.add_argument("--judge-jsonl", default=None,
                   help="Override path to llm_judge.jsonl.")
    p.add_argument("--metrics-jsonl", default=None,
                   help="Override path to training metrics.jsonl.")
    p.add_argument("--sim-ab-json", default=None,
                   help="Override path to sim_ab.json (P2). Missing is OK.")
    p.add_argument("--out-json", default=None,
                   help="Where to write v3_scorecard.json (default: <ckpt>/v3_scorecard.json).")
    p.add_argument("--exit-on-fail", action="store_true",
                   help="Exit non-zero (2) if overall == FAIL. Useful in CI.")
    p.add_argument("--exit-on-warn", action="store_true",
                   help="Exit non-zero (3) if overall == WARN or FAIL.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    ckpt_dir = Path(args.ckpt_dir)
    retrieval_p = Path(args.retrieval_json) if args.retrieval_json else ckpt_dir / "retrieval_margin.json"
    judge_p = Path(args.judge_jsonl) if args.judge_jsonl else ckpt_dir / "llm_judge.jsonl"
    metrics_p = Path(args.metrics_jsonl) if args.metrics_jsonl else ckpt_dir / "metrics.jsonl"
    sim_p = Path(args.sim_ab_json) if args.sim_ab_json else ckpt_dir / "sim_ab.json"
    out_p = Path(args.out_json) if args.out_json else ckpt_dir / "v3_scorecard.json"

    print(f"checkpoint:       {ckpt_dir}")
    print(f"  retrieval_json: {retrieval_p}  [{'exists' if retrieval_p.exists() else 'MISSING'}]")
    print(f"  judge_jsonl:    {judge_p}      [{'exists' if judge_p.exists() else 'MISSING'}]")
    print(f"  metrics_jsonl:  {metrics_p}    [{'exists' if metrics_p.exists() else 'MISSING'}]")
    print(f"  sim_ab_json:    {sim_p}        [{'exists' if sim_p.exists() else 'not present (P2)'}]")

    values: dict[str, Any] = {}
    values.update(_read_retrieval(retrieval_p))
    values.update(_read_judge(judge_p))
    values.update(_read_training_metrics(metrics_p))
    values.update(_read_sim_ab(sim_p))

    rows = _build_metric_rows(values)
    overall = _overall_verdict(rows)

    scorecard = Scorecard(
        checkpoint=str(ckpt_dir),
        overall=overall,
        metrics=[asdict(r) for r in rows],
        sources={
            "retrieval_json": str(retrieval_p),
            "judge_jsonl": str(judge_p),
            "metrics_jsonl": str(metrics_p),
            "sim_ab_json": str(sim_p),
        },
        config={
            "judge_n": values.get("judge_n", 0),
            "judge_n_template": values.get("judge_n_template", 0),
            "training_step": values.get("training_step"),
            "sim_present": values.get("sim_present", False),
            "sim_n_episodes_per_arm": values.get("sim_n_episodes", 0),
            "sim_per_suite": values.get("sim_per_suite", {}),
        },
    )

    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(asdict(scorecard), indent=2))

    # Console summary
    print()
    print("=" * 78)
    print(f"V3 LIBERO SCORECARD                       overall = {overall}")
    print("=" * 78)
    print(f"  {'metric':<40s}  {'value':>9s}  {'pass>=':>9s}  {'warn>=':>9s}  verdict  req")
    print(f"  {'-'*40}  {'-'*9}  {'-'*9}  {'-'*9}  -------  ---")
    for r in rows:
        if r.value is None:
            v_str = "    N/A"
        else:
            v_str = f"{r.value:9.4f}" if abs(r.value) < 100 else f"{r.value:9.2f}"
        req = "yes" if r.required_for_overall else "  -"
        cmp_pass = f"{r.threshold_pass:9.4f}"
        cmp_warn = f"{r.threshold_warn:9.4f}"
        print(f"  {r.name:<40s}  {v_str}  {cmp_pass}  {cmp_warn}  {r.verdict:<7s}  {req}")
    print()
    per_suite = values.get("sim_per_suite") or {}
    if per_suite:
        print("  Per-suite sim success-rates:")
        suite_names = sorted(per_suite.keys())
        print(f"    {'suite':<10s}  {'baseline':>10s}  {'correct':>10s}  {'wrong':>10s}  {'c-w':>8s}")
        for sn in suite_names:
            arms = per_suite[sn]
            def _fmt(v): return f"{v:9.3f}" if isinstance(v, (int, float)) else "      N/A"
            c = arms.get("correct"); w = arms.get("wrong"); b = arms.get("baseline")
            gap = (
                f"{c - w:+8.3f}"
                if isinstance(c, (int, float)) and isinstance(w, (int, float))
                else "    N/A"
            )
            print(f"    {sn:<10s}  {_fmt(b)}  {_fmt(c)}  {_fmt(w)}  {gap}")
        print()
    print(f"  -> {out_p}")
    print()
    if overall == "PASS":
        print("  V3 is REAL: retrieval margin + grounding + (sim/anti-template) all PASS.")
    elif overall == "WARN":
        print("  V3 is borderline: some required gate is WARN/NA.")
    else:
        print("  V3 is FAIL: at least one required gate is below the warn band.")

    if args.exit_on_fail and overall == "FAIL":
        return 2
    if args.exit_on_warn and overall in ("FAIL", "WARN"):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
