#!/usr/bin/env python
"""Aggregate auto metrics + LLM-judge rubric into final eval scores.

Inputs:
    - ``eval_cases.jsonl``    (frozen hypotheses)
    - ``panel_rows.jsonl``    (baseline/edited/control evidence + AR deltas)
    - ``judge_rows.jsonl``    (constrained LLM rubric scores; optional)

Outputs:
    - ``scores_by_case.jsonl``  per-case row with auto metrics, judge metrics,
                                and composite faithfulness score.
    - ``scores.json``           summary aggregates (means, std, per-stratum
                                breakdowns, counts of failures).

Auto metrics (paper-grade, deterministic)
-----------------------------------------

For each case, given the panel-row vectors and texts:

    direction_match
        +1 if AR-edited reconstruction has lower MSE to ``h_edit`` than to
        ``h``, normalized by the same difference for the control. This is a
        signed scalar in [-1, 1] -- positive means the edit moved the
        explanation in a way the AR can pick up; control adjusts for noise.

    normalized_effect_size
        Cohen-d-style scaled difference between AR-edited and AR-baseline MSE,
        divided by AR-baseline MSE. Larger = bigger explanation shift.

    seed_stability
        Mean pairwise word-level overlap among ``seed_stability_texts``;
        lower stability + high direction_match is the desired profile (the
        edit moves things, but the explanation is otherwise stable).

    confabulation_score (auto)
        Fraction of evidence quotes the judge listed that we could verify
        verbatim against the source text. (Computed from judge rows.)

Composite
---------

    composite = w_auto * auto_score + w_judge * judge_score

where ``auto_score`` is the L2-clamped mean of the deterministic metrics
mapped into [0, 1], and ``judge_score`` is the [0,1]-normalized mean of the
rubric integers.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

# Local-import rubric for normalization.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from rubric import RUBRIC_DEFINITIONS  # type: ignore  # noqa: E402

logger = logging.getLogger("nla.eval.score")


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Auto metrics
# ---------------------------------------------------------------------------

def _direction_match(panel_row: dict[str, Any]) -> float | None:
    """Did AR-edited reconstruct h_edit better than h, vs. control?

    Returns:
        Float in [-1, 1] roughly (clamped), or None if AR not present.
    """
    inp = panel_row.get("auto_metrics_inputs", {})
    if not inp.get("ar_present"):
        return None
    base = inp.get("ar_baseline_mse")
    edit = inp.get("ar_edited_mse")
    ctrl = inp.get("ar_control_mse")
    if base is None or edit is None or ctrl is None:
        return None
    # The intuition: if the edit moved AR meaningfully (edit reconstructs h_edit
    # well, baseline reconstructs h well), and the control moved AR less, then
    # direction_match > 0.
    edit_gain = base - edit
    ctrl_gain = base - ctrl
    denom = max(abs(edit_gain), abs(ctrl_gain), 1e-9)
    val = (edit_gain - ctrl_gain) / denom
    return max(-1.0, min(1.0, val))


def _normalized_effect_size(panel_row: dict[str, Any]) -> float | None:
    inp = panel_row.get("auto_metrics_inputs", {})
    if not inp.get("ar_present"):
        return None
    base = inp.get("ar_baseline_mse")
    edit = inp.get("ar_edited_mse")
    if base is None or edit is None or base <= 0:
        return None
    return (edit - base) / base


def _seed_stability(panel_row: dict[str, Any]) -> float | None:
    """Mean pairwise word-level Jaccard among the stability texts."""
    texts = panel_row.get("seed_stability_texts") or []
    if len(texts) < 2:
        return None
    sims: list[float] = []
    sets = [set(t.lower().split()) for t in texts]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            if not a and not b:
                sims.append(1.0)
                continue
            inter = len(a & b)
            union = max(1, len(a | b))
            sims.append(inter / union)
    if not sims:
        return None
    return sum(sims) / len(sims)


def _confabulation_score(judge_rows_for_case: list[dict[str, Any]]) -> float | None:
    """Mean across judges of (fraction of evidence quotes that survived
    verbatim verification)."""
    if not judge_rows_for_case:
        return None
    fracs: list[float] = []
    for row in judge_rows_for_case:
        spans = row.get("evidence_spans") or []
        warns = row.get("_warnings") or []
        n_dropped = sum(1 for w in warns if "not in" in w and "dropped" in w)
        n_kept = len(spans)
        denom = n_kept + n_dropped
        if denom == 0:
            continue
        fracs.append(n_kept / denom)
    if not fracs:
        return None
    return sum(fracs) / len(fracs)


# ---------------------------------------------------------------------------
# Judge-side aggregation
# ---------------------------------------------------------------------------

def _judge_score_01(judge_rows_for_case: list[dict[str, Any]]) -> float | None:
    """Mean rubric score in [0, 1] across judges and dimensions."""
    if not judge_rows_for_case:
        return None
    norm_means: list[float] = []
    for row in judge_rows_for_case:
        per_dim: list[float] = []
        for k, dim in RUBRIC_DEFINITIONS.items():
            v = row.get(k)
            if v is None:
                continue
            span = max(1, dim.max_val - dim.min_val)
            per_dim.append((v - dim.min_val) / span)
        if per_dim:
            norm_means.append(sum(per_dim) / len(per_dim))
    if not norm_means:
        return None
    return sum(norm_means) / len(norm_means)


def _judge_agreement(judge_rows_for_case: list[dict[str, Any]]) -> float | None:
    """Inter-judge agreement: mean(1 - |a-b|/range) over rubric dims, pairs."""
    if len(judge_rows_for_case) < 2:
        return None
    pair_scores: list[float] = []
    for i in range(len(judge_rows_for_case)):
        for j in range(i + 1, len(judge_rows_for_case)):
            a = judge_rows_for_case[i]
            b = judge_rows_for_case[j]
            per_dim: list[float] = []
            for k, dim in RUBRIC_DEFINITIONS.items():
                if k not in a or k not in b:
                    continue
                span = max(1, dim.max_val - dim.min_val)
                per_dim.append(1.0 - abs(a[k] - b[k]) / span)
            if per_dim:
                pair_scores.append(sum(per_dim) / len(per_dim))
    if not pair_scores:
        return None
    return sum(pair_scores) / len(pair_scores)


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

def _auto_score_01(
    direction_match: float | None,
    normalized_effect_size: float | None,
    seed_stability: float | None,
) -> float | None:
    """Map deterministic auto metrics into a single [0, 1] aggregate.

    - direction_match in [-1, 1] -> [0, 1] via (x+1)/2
    - normalized_effect_size: large absolute values -> 1, near zero -> 0
      via a soft tanh squash on |x|.
    - seed_stability in [0, 1] is itself a goodness signal.
    """
    parts: list[float] = []
    if direction_match is not None:
        parts.append((direction_match + 1.0) / 2.0)
    if normalized_effect_size is not None:
        parts.append(math.tanh(abs(normalized_effect_size)))
    if seed_stability is not None:
        parts.append(seed_stability)
    if not parts:
        return None
    return sum(parts) / len(parts)


def _composite(
    auto_01: float | None,
    judge_01: float | None,
    *,
    w_auto: float,
    w_judge: float,
) -> float | None:
    if auto_01 is None and judge_01 is None:
        return None
    if auto_01 is None:
        return float(judge_01)
    if judge_01 is None:
        return float(auto_01)
    total = w_auto + w_judge
    return (w_auto * auto_01 + w_judge * judge_01) / total


# ---------------------------------------------------------------------------
# Aggregation summary
# ---------------------------------------------------------------------------

def _safe_mean(xs: Iterable[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return statistics.fmean(vals) if vals else None


def _safe_stdev(xs: Iterable[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return statistics.stdev(vals) if len(vals) > 1 else None


def _summarize(rows: list[dict[str, Any]], *, group: str | None = None) -> dict[str, Any]:
    if group is None:
        return {
            "n": len(rows),
            "auto_score_01": {
                "mean": _safe_mean(r.get("auto_score_01") for r in rows),
                "std": _safe_stdev(r.get("auto_score_01") for r in rows),
            },
            "judge_score_01": {
                "mean": _safe_mean(r.get("judge_score_01") for r in rows),
                "std": _safe_stdev(r.get("judge_score_01") for r in rows),
            },
            "composite": {
                "mean": _safe_mean(r.get("composite") for r in rows),
                "std": _safe_stdev(r.get("composite") for r in rows),
            },
            "direction_match": {
                "mean": _safe_mean(r.get("direction_match") for r in rows),
            },
            "judge_agreement": {
                "mean": _safe_mean(r.get("judge_agreement") for r in rows),
            },
            "confabulation_score": {
                "mean": _safe_mean(r.get("confabulation_score") for r in rows),
            },
            "n_failed": sum(1 for r in rows if r.get("composite") is None),
        }
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[str(r.get(group))].append(r)
    return {k: _summarize(v) for k, v in by.items()}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cases", required=True, help="eval_cases.jsonl")
    p.add_argument("--panel", required=True, help="panel_rows.jsonl")
    p.add_argument(
        "--judge",
        default=None,
        help="judge_rows.jsonl (optional). If omitted, only auto metrics are produced.",
    )
    p.add_argument("--out-by-case", required=True, help="scores_by_case.jsonl path")
    p.add_argument("--out-summary", required=True, help="scores.json path")
    p.add_argument("--w-auto", type=float, default=0.7)
    p.add_argument("--w-judge", type=float, default=0.3)
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    cases = _load_jsonl(Path(args.cases))
    panel = _load_jsonl(Path(args.panel))
    judges = _load_jsonl(Path(args.judge)) if args.judge else []

    panel_by_id = {r["case_id"]: r for r in panel}
    judges_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for jr in judges:
        judges_by_id[jr["case_id"]].append(jr)

    out_rows: list[dict[str, Any]] = []
    for case in cases:
        cid = case["case_id"]
        prow = panel_by_id.get(cid)
        if prow is None:
            logger.warning("No panel row for %s; skipping", cid)
            continue
        jrows = judges_by_id.get(cid, [])

        dm = _direction_match(prow)
        es = _normalized_effect_size(prow)
        ss = _seed_stability(prow)
        conf = _confabulation_score(jrows)
        judge_01 = _judge_score_01(jrows)
        agreement = _judge_agreement(jrows)
        auto_01 = _auto_score_01(dm, es, ss)
        composite = _composite(auto_01, judge_01, w_auto=args.w_auto, w_judge=args.w_judge)

        out_rows.append(
            {
                "case_id": cid,
                "position_type": case.get("position_type"),
                "edit_kind": prow.get("intervention_spec", {}).get("edit_kind"),
                "direction_match": dm,
                "normalized_effect_size": es,
                "seed_stability": ss,
                "confabulation_score": conf,
                "auto_score_01": auto_01,
                "judge_score_01": judge_01,
                "judge_agreement": agreement,
                "composite": composite,
                "n_judges": len(jrows),
            }
        )

    out_by_case_path = Path(args.out_by_case)
    out_by_case_path.parent.mkdir(parents=True, exist_ok=True)
    with out_by_case_path.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    summary: dict[str, Any] = {
        "n_cases": len(out_rows),
        "weights": {"auto": args.w_auto, "judge": args.w_judge},
        "overall": _summarize(out_rows),
        "by_position_type": _summarize(out_rows, group="position_type"),
        "by_edit_kind": _summarize(out_rows, group="edit_kind"),
    }
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary).write_text(json.dumps(summary, indent=2))

    logger.info("Wrote %d per-case rows to %s", len(out_rows), out_by_case_path)
    logger.info("Wrote summary to %s", args.out_summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
