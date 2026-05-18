#!/usr/bin/env python
"""Build a "what to improve" markdown report from an SFT checkpoint's eval outputs.

Reads every known eval artifact that a post-SFT eval run may have produced
(scorecard, extraction_diag, retrieval_margin, sim_ab, llm_judge.jsonl,
av_samples.jsonl, metrics.jsonl) and synthesises a single markdown digest:

  - Sections by axis: reconstruction quality, retrieval / discrimination,
    AV grounding, behavioral signal, position-type breakdown.
  - Each finding labelled IMPROVED / STABLE / REGRESSED / UNCHECKED when a
    --baseline-dir is supplied, otherwise STRONG / WEAK / UNKNOWN.
  - A ranked "If you want to improve X, try Y" action list at the end that
    references actual flags in ``scripts/training/run_sft.py`` and
    ``scripts/training/run_grpo.py``.

Every input file is optional - missing files become one-line "missing" notes
rather than crashes. Pure CPU, no LLM / GPU dependency.

Usage:
    python scripts/eval/build_improvements_report.py \
        --sft-dir data/sft/libero_4suite_v3 \
        [--baseline-dir data/sft/libero_4suite_v2] \
        [--output PATH]

The default output path is ``<sft-dir>/improvements.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any


logger = logging.getLogger("build_improvements_report")


# ---------------------------------------------------------------------------
# File loading (every file is optional)
# ---------------------------------------------------------------------------


KNOWN_FILES = (
    "v3_scorecard.json",          # legacy name used by current eval pipeline
    "scorecard.json",             # in case the pipeline ever renames it
    "extraction_diag.json",
    "v4_extraction_scorecard.json",
    "retrieval_margin.json",
    "sim_ab.json",
    "llm_judge.jsonl",
    "av_samples.jsonl",
    "metrics.jsonl",
)


def _resolve(sft_dir: Path, name: str) -> Path | None:
    """Return the first existing instance of `name` in sft_dir or post_sft_eval/."""
    candidates = [sft_dir / name, sft_dir / "post_sft_eval" / name]
    for c in candidates:
        if c.exists():
            return c
    return None


def _read_json(path: Path | None) -> dict | None:
    if path is None:
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        logger.warning("failed to parse %s: %s", path, e)
        return None


def _read_jsonl(path: Path | None) -> list[dict] | None:
    if path is None:
        return None
    rows: list[dict] = []
    try:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return rows
    except Exception as e:
        logger.warning("failed to read jsonl %s: %s", path, e)
        return None


def load_bundle(sft_dir: Path) -> dict[str, Any]:
    """Load every known eval artifact present, return {key: data_or_None, _paths}."""
    bundle: dict[str, Any] = {"_paths": {}, "_missing": []}

    scorecard_path = _resolve(sft_dir, "v3_scorecard.json") or _resolve(sft_dir, "scorecard.json")
    bundle["scorecard"] = _read_json(scorecard_path)
    bundle["_paths"]["scorecard"] = scorecard_path
    if scorecard_path is None:
        bundle["_missing"].append("scorecard (v3_scorecard.json | scorecard.json)")

    for key, name in (
        ("extraction_diag", "extraction_diag.json"),
        ("extraction_scorecard", "v4_extraction_scorecard.json"),
        ("retrieval", "retrieval_margin.json"),
        ("sim_ab", "sim_ab.json"),
    ):
        path = _resolve(sft_dir, name)
        bundle[key] = _read_json(path)
        bundle["_paths"][key] = path
        if path is None:
            bundle["_missing"].append(name)

    for key, name in (
        ("llm_judge", "llm_judge.jsonl"),
        ("av_samples", "av_samples.jsonl"),
        ("metrics", "metrics.jsonl"),
    ):
        path = _resolve(sft_dir, name)
        bundle[key] = _read_jsonl(path)
        bundle["_paths"][key] = path
        if path is None:
            bundle["_missing"].append(name)

    return bundle


# ---------------------------------------------------------------------------
# Derived metric extractors. All return Optional[float|dict] and never raise.
# ---------------------------------------------------------------------------


def _final_row(metrics: list[dict] | None) -> dict | None:
    if not metrics:
        return None
    for r in reversed(metrics):
        if r.get("phase") == "final":
            return r
    return None


def _last_train_row(metrics: list[dict] | None) -> dict | None:
    if not metrics:
        return None
    for r in reversed(metrics):
        if r.get("phase") == "train":
            return r
    return metrics[-1] if metrics else None


def _scorecard_metric(scorecard: dict | None, name: str) -> dict | None:
    if not scorecard:
        return None
    for m in scorecard.get("metrics", []) or []:
        if m.get("name") == name:
            return m
    return None


def _per_position_from_final(final_row: dict | None, prefix: str) -> dict[str, float]:
    """Pull keys like 'fve/position=image_patch' under a given prefix.

    `prefix` may be e.g. 'fve' or 'closed_greedy/cosine'. Returns
    {ptype: value} dict for every position found.
    """
    out: dict[str, float] = {}
    if not final_row:
        return out
    needle = f"{prefix}/position="
    for k, v in final_row.items():
        if isinstance(k, str) and k.startswith(needle) and isinstance(v, (int, float)):
            out[k[len(needle):]] = float(v)
    return out


def _judge_pass_rates(judge_rows: list[dict] | None) -> dict[str, Any]:
    """Compute pass rates for the three judge axes, segmented by variant_id."""
    out: dict[str, Any] = {"n": 0, "by_variant": {}}
    if not judge_rows:
        return out
    by_variant: dict[str, dict[str, list[int]]] = {}
    for r in judge_rows:
        v = str(r.get("variant_id") or "unknown")
        d = by_variant.setdefault(v, {"grounding": [], "appropriateness": [], "template_distinguishable": []})
        for axis in d:
            try:
                passed = bool(r.get(axis, {}).get("passed"))
                d[axis].append(1 if passed else 0)
            except Exception:
                pass
    out["n"] = len(judge_rows)
    for v, d in by_variant.items():
        out["by_variant"][v] = {
            axis: (sum(vals) / len(vals) if vals else None)
            for axis, vals in d.items()
        } | {"_n": min((len(vals) for vals in d.values()), default=0)}
    return out


def _av_diversity(av_rows: list[dict] | None) -> dict[str, Any]:
    """Cheap surface-form diversity of generated AV bullets.

    We measure: distinct first-line ratio, distinct exact-generated ratio,
    and mean / median cosine and MSE (teacher-forced vs closed-loop) per
    position_type. This is good enough to flag template collapse without
    pulling in an embedder.
    """
    out: dict[str, Any] = {"n_rows": 0, "by_ptype": {}}
    if not av_rows:
        return out
    out["n_rows"] = len(av_rows)
    by_ptype: dict[str, list[dict]] = {}
    for r in av_rows:
        pt = str(r.get("position_type") or "unknown")
        by_ptype.setdefault(pt, []).append(r)
    for pt, rows in by_ptype.items():
        gens = [str(r.get("generated") or "") for r in rows]
        firsts = [g.split("\n", 1)[0].strip().lower() for g in gens]
        n = len(rows)
        distinct_full = len({g.strip() for g in gens if g.strip()})
        distinct_first = len({f for f in firsts if f})
        def _mean(seq: list[float]) -> float | None:
            seq = [x for x in seq if isinstance(x, (int, float)) and not math.isnan(x)]
            return (sum(seq) / len(seq)) if seq else None
        tf_cos = _mean([r.get("tf_cosine") for r in rows])
        tf_mse = _mean([r.get("tf_mse") for r in rows])
        cl_cos = _mean([r.get("cl_cosine") for r in rows])
        cl_mse = _mean([r.get("cl_mse") for r in rows])
        out["by_ptype"][pt] = {
            "n": n,
            "distinct_generated": distinct_full,
            "distinct_first_line": distinct_first,
            "distinct_first_ratio": (distinct_first / n) if n else None,
            "tf_cosine_mean": tf_cos,
            "tf_mse_mean": tf_mse,
            "cl_cosine_mean": cl_cos,
            "cl_mse_mean": cl_mse,
            "cl_cosine_drop_vs_tf": (
                (tf_cos - cl_cos) if (tf_cos is not None and cl_cos is not None) else None
            ),
        }
    return out


def derive(bundle: dict[str, Any]) -> dict[str, Any]:
    """Compute the headline metrics used by both the report and the diff."""
    scorecard = bundle.get("scorecard")
    extraction = bundle.get("extraction_diag")
    retrieval = bundle.get("retrieval")
    sim_ab = bundle.get("sim_ab")
    judge = bundle.get("llm_judge")
    av = bundle.get("av_samples")
    metrics = bundle.get("metrics")

    final = _final_row(metrics)
    last_train = _last_train_row(metrics)

    d: dict[str, Any] = {
        "overall_verdict": (scorecard or {}).get("overall"),
        "training_step": ((scorecard or {}).get("config") or {}).get("training_step"),

        "fve_total": (final or {}).get("fve"),
        "mse_total": (final or {}).get("mse"),
        "cosine_total": (final or {}).get("cosine"),
        "closed_greedy_cosine_total": (final or {}).get("closed_greedy/cosine"),
        "closed_greedy_fve_total": (final or {}).get("closed_greedy/fve"),

        "fve_by_ptype": _per_position_from_final(final, "fve"),
        "cosine_by_ptype": _per_position_from_final(final, "cosine"),
        "mse_by_ptype": _per_position_from_final(final, "mse"),
        "closed_greedy_cosine_by_ptype": _per_position_from_final(final, "closed_greedy/cosine"),
        "closed_greedy_fve_by_ptype": _per_position_from_final(final, "closed_greedy/fve"),

        "closed_loop_cosine_gap": (
            ((final or {}).get("cosine") - (final or {}).get("closed_greedy/cosine"))
            if (final and final.get("cosine") is not None and final.get("closed_greedy/cosine") is not None)
            else None
        ),

        "ar_nce_last_train": (last_train or {}).get("ar_nce"),
        "ar_mse_last_train": (last_train or {}).get("ar_mse"),
        "ce_last_train": (last_train or {}).get("ce"),
        "loss_last_train": (last_train or {}).get("loss"),
        "p_av_last_train": (last_train or {}).get("p_av"),

        "retrieval_margin": (retrieval or {}).get("margin"),
        "retrieval_at_1": (retrieval or {}).get("retrieval_at_1"),
        "retrieval_at_5": (retrieval or {}).get("retrieval_at_5"),
        "retrieval_at_10": (retrieval or {}).get("retrieval_at_10"),
        "retrieval_by_position": (retrieval or {}).get("by_position", {}) or {},

        "hard_neg_by_ptype": (((extraction or {}).get("hard_negatives") or {}).get("by_ptype")) or {},
        "suite_probe_by_ptype": (extraction or {}).get("suite_probe") or {},
        "episode_cosine_gap_by_ptype": (extraction or {}).get("episode_cosine_gap") or {},

        "sim_correct_success": (sim_ab or {}).get("correct_success_mean"),
        "sim_wrong_success": (sim_ab or {}).get("wrong_success_mean"),
        "sim_baseline_success": (sim_ab or {}).get("baseline_success_mean"),
        "sim_correct_minus_wrong": (sim_ab or {}).get("correct_minus_wrong"),
        "sim_correct_minus_baseline": (sim_ab or {}).get("correct_minus_baseline"),

        "judge": _judge_pass_rates(judge),
        "av_diversity": _av_diversity(av),
    }

    sc = scorecard or {}
    d["scorecard_metrics"] = {m.get("name"): m for m in (sc.get("metrics") or []) if m.get("name")}
    return d


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------


def _classify_diff(
    current: float | None,
    baseline: float | None,
    *,
    higher_is_better: bool = True,
    eps_abs: float = 0.005,
    eps_rel: float = 0.05,
) -> tuple[str, str]:
    """Return (label, formatted_delta) for a diff against baseline.

    label ∈ {IMPROVED, STABLE, REGRESSED, UNCHECKED}.
    eps_abs and eps_rel define the STABLE band: |delta| <= max(eps_abs, eps_rel * |baseline|).
    """
    if current is None or baseline is None:
        return "UNCHECKED", "n/a"
    try:
        delta = float(current) - float(baseline)
    except Exception:
        return "UNCHECKED", "n/a"
    band = max(eps_abs, eps_rel * abs(float(baseline)))
    if abs(delta) <= band:
        return "STABLE", f"Δ{delta:+.4f}"
    if (delta > 0) == higher_is_better:
        return "IMPROVED", f"Δ{delta:+.4f}"
    return "REGRESSED", f"Δ{delta:+.4f}"


def _classify_absolute(
    value: float | None,
    *,
    strong_threshold: float | None = None,
    weak_threshold: float | None = None,
    higher_is_better: bool = True,
) -> str:
    """Return STRONG / WEAK / UNKNOWN given absolute thresholds.

    When higher_is_better=True: STRONG if value >= strong_threshold, WEAK if <= weak_threshold.
    Otherwise inverted. Anything in between, or with missing thresholds, is WEAK
    (we err toward "the user should look at this").
    """
    if value is None:
        return "UNKNOWN"
    if strong_threshold is None and weak_threshold is None:
        return "UNKNOWN"
    if higher_is_better:
        if strong_threshold is not None and value >= strong_threshold:
            return "STRONG"
        if weak_threshold is not None and value <= weak_threshold:
            return "WEAK"
        return "WEAK"
    if strong_threshold is not None and value <= strong_threshold:
        return "STRONG"
    if weak_threshold is not None and value >= weak_threshold:
        return "WEAK"
    return "WEAK"


def _fmt_num(v: Any, places: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int,)):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return str(v)
        return f"{v:.{places}f}"
    return str(v)


def _fmt_pct(v: Any, places: int = 1) -> str:
    if v is None:
        return "—"
    try:
        return f"{100.0 * float(v):.{places}f}%"
    except Exception:
        return str(v)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _line_for(
    label: str,
    current: float | None,
    baseline: float | None,
    *,
    fmt: str = "num",
    higher_is_better: bool = True,
    strong: float | None = None,
    weak: float | None = None,
    eps_abs: float = 0.005,
    eps_rel: float = 0.05,
    has_baseline: bool,
    extra: str = "",
) -> str:
    """Render one bullet describing a single metric."""
    if fmt == "pct":
        cur_s = _fmt_pct(current)
        base_s = _fmt_pct(baseline)
    else:
        cur_s = _fmt_num(current)
        base_s = _fmt_num(baseline)

    if has_baseline:
        verdict, delta_s = _classify_diff(
            current, baseline,
            higher_is_better=higher_is_better,
            eps_abs=eps_abs, eps_rel=eps_rel,
        )
        if current is None:
            return f"- **{label}**: UNCHECKED (not in current run)"
        return f"- **{label}**: {cur_s} (baseline: {base_s}, {delta_s}) → **{verdict}**{(' — ' + extra) if extra else ''}"
    verdict = _classify_absolute(
        current,
        strong_threshold=strong, weak_threshold=weak,
        higher_is_better=higher_is_better,
    )
    return f"- **{label}**: {cur_s} → **{verdict}**{(' — ' + extra) if extra else ''}"


def build_reconstruction_section(
    d: dict[str, Any],
    b: dict[str, Any] | None,
) -> tuple[str, list[dict]]:
    has_b = b is not None
    b = b or {}
    findings: list[dict] = []
    lines: list[str] = ["## Reconstruction quality", ""]

    if d.get("fve_total") is None and not d.get("fve_by_ptype"):
        lines.append("- _missing_: no `metrics.jsonl` `phase=final` row found.")
        return "\n".join(lines) + "\n", findings

    lines.append(_line_for(
        "FVE (teacher-forced, total)", d.get("fve_total"), b.get("fve_total"),
        higher_is_better=True, strong=0.3, weak=0.0,
        eps_abs=0.02, eps_rel=0.10,
        has_baseline=has_b,
        extra="negative FVE means model is worse than mean-predictor; positive ≥0.3 is healthy",
    ))
    lines.append(_line_for(
        "Cosine (teacher-forced, total)", d.get("cosine_total"), b.get("cosine_total"),
        higher_is_better=True, strong=0.55, weak=0.4,
        eps_abs=0.01, eps_rel=0.05,
        has_baseline=has_b,
    ))
    lines.append(_line_for(
        "MSE (teacher-forced, total)", d.get("mse_total"), b.get("mse_total"),
        higher_is_better=False, strong=200.0, weak=400.0,
        eps_abs=5.0, eps_rel=0.05,
        has_baseline=has_b,
    ))
    lines.append(_line_for(
        "Closed-loop greedy cosine (total)",
        d.get("closed_greedy_cosine_total"), b.get("closed_greedy_cosine_total"),
        higher_is_better=True, strong=0.55, weak=0.4,
        eps_abs=0.01, eps_rel=0.05,
        has_baseline=has_b,
    ))
    lines.append(_line_for(
        "Closed-loop cosine gap (TF − CL)",
        d.get("closed_loop_cosine_gap"), b.get("closed_loop_cosine_gap"),
        higher_is_better=False, strong=0.02, weak=0.10,
        eps_abs=0.01, eps_rel=0.10,
        has_baseline=has_b,
        extra="small gap = sampling stays on-manifold",
    ))

    cur_pt = d.get("fve_by_ptype") or {}
    base_pt = b.get("fve_by_ptype") or {}
    if cur_pt:
        lines.append("")
        lines.append("Per-position FVE (teacher-forced):")
        lines.append("")
        lines.append("| Position | Current | Baseline | Δ | Verdict |")
        lines.append("|---|---:|---:|---:|---|")
        for pt in sorted(cur_pt):
            cur_v = cur_pt.get(pt)
            base_v = base_pt.get(pt) if has_b else None
            if has_b:
                v_label, delta_s = _classify_diff(cur_v, base_v, higher_is_better=True, eps_abs=0.02, eps_rel=0.10)
            else:
                v_label = _classify_absolute(cur_v, strong_threshold=0.3, weak_threshold=0.0)
                delta_s = "—"
            lines.append(f"| `{pt}` | {_fmt_num(cur_v)} | {_fmt_num(base_v)} | {delta_s} | {v_label} |")
            findings.append({
                "axis": "reconstruction", "metric": f"fve_{pt}",
                "value": cur_v, "baseline": base_v, "verdict": v_label,
            })

    findings.append({
        "axis": "reconstruction", "metric": "fve_total",
        "value": d.get("fve_total"), "baseline": b.get("fve_total"),
        "verdict": (
            _classify_diff(d.get("fve_total"), b.get("fve_total"), higher_is_better=True, eps_abs=0.02, eps_rel=0.10)[0]
            if has_b else _classify_absolute(d.get("fve_total"), strong_threshold=0.3, weak_threshold=0.0)
        ),
    })
    findings.append({
        "axis": "reconstruction", "metric": "closed_loop_cosine_gap",
        "value": d.get("closed_loop_cosine_gap"), "baseline": b.get("closed_loop_cosine_gap"),
        "verdict": (
            _classify_diff(d.get("closed_loop_cosine_gap"), b.get("closed_loop_cosine_gap"), higher_is_better=False, eps_abs=0.01, eps_rel=0.10)[0]
            if has_b else _classify_absolute(d.get("closed_loop_cosine_gap"), strong_threshold=0.02, weak_threshold=0.10, higher_is_better=False)
        ),
    })

    return "\n".join(lines) + "\n", findings


def build_retrieval_section(
    d: dict[str, Any], b: dict[str, Any] | None,
) -> tuple[str, list[dict]]:
    has_b = b is not None
    b = b or {}
    findings: list[dict] = []
    lines: list[str] = ["## Retrieval / discrimination", ""]

    have_any = (
        d.get("retrieval_margin") is not None
        or d.get("retrieval_at_1") is not None
        or d.get("ar_nce_last_train") is not None
        or d.get("hard_neg_by_ptype")
    )
    if not have_any:
        lines.append("- _missing_: no `retrieval_margin.json`, `metrics.jsonl` or `extraction_diag.json` data.")
        return "\n".join(lines) + "\n", findings

    lines.append(_line_for(
        "Retrieval margin (matched − cross cos)",
        d.get("retrieval_margin"), b.get("retrieval_margin"),
        higher_is_better=True, strong=0.05, weak=0.02,
        eps_abs=0.005, eps_rel=0.05,
        has_baseline=has_b,
    ))
    lines.append(_line_for(
        "Retrieval@1", d.get("retrieval_at_1"), b.get("retrieval_at_1"),
        fmt="pct", higher_is_better=True, strong=0.25, weak=0.15,
        eps_abs=0.01, eps_rel=0.05,
        has_baseline=has_b,
    ))
    lines.append(_line_for(
        "Retrieval@5", d.get("retrieval_at_5"), b.get("retrieval_at_5"),
        fmt="pct", higher_is_better=True, strong=0.55, weak=0.40,
        eps_abs=0.01, eps_rel=0.05,
        has_baseline=has_b,
    ))
    lines.append(_line_for(
        "Retrieval@10", d.get("retrieval_at_10"), b.get("retrieval_at_10"),
        fmt="pct", higher_is_better=True, strong=0.70, weak=0.50,
        eps_abs=0.01, eps_rel=0.05,
        has_baseline=has_b,
    ))
    lines.append(_line_for(
        "AR-NCE loss (last train row)",
        d.get("ar_nce_last_train"), b.get("ar_nce_last_train"),
        higher_is_better=False, strong=1.5, weak=3.0,
        eps_abs=0.05, eps_rel=0.05,
        has_baseline=has_b,
        extra="lower = positive pair separates from in-batch negatives",
    ))

    hn_cur = d.get("hard_neg_by_ptype") or {}
    hn_base = b.get("hard_neg_by_ptype") or {}
    if hn_cur:
        lines.append("")
        lines.append("Hard-negative tightness (median cos_top1, lower = better discrimination):")
        lines.append("")
        lines.append("| Position | Current | Baseline | Δ | Verdict |")
        lines.append("|---|---:|---:|---:|---|")
        for pt in sorted(hn_cur):
            cur_v = hn_cur.get(pt, {}).get("median_cos_top1")
            base_v = hn_base.get(pt, {}).get("median_cos_top1") if has_b else None
            if has_b:
                v_label, delta_s = _classify_diff(cur_v, base_v, higher_is_better=False, eps_abs=0.005, eps_rel=0.02)
            else:
                v_label = _classify_absolute(cur_v, strong_threshold=0.95, weak_threshold=0.97, higher_is_better=False)
                delta_s = "—"
            lines.append(f"| `{pt}` | {_fmt_num(cur_v)} | {_fmt_num(base_v)} | {delta_s} | {v_label} |")
            findings.append({
                "axis": "retrieval", "metric": f"hard_neg_{pt}",
                "value": cur_v, "baseline": base_v, "verdict": v_label,
            })

    findings.append({
        "axis": "retrieval", "metric": "retrieval_margin",
        "value": d.get("retrieval_margin"), "baseline": b.get("retrieval_margin"),
        "verdict": (
            _classify_diff(d.get("retrieval_margin"), b.get("retrieval_margin"), higher_is_better=True, eps_abs=0.005, eps_rel=0.05)[0]
            if has_b else _classify_absolute(d.get("retrieval_margin"), strong_threshold=0.05, weak_threshold=0.02)
        ),
    })
    findings.append({
        "axis": "retrieval", "metric": "retrieval_at_1",
        "value": d.get("retrieval_at_1"), "baseline": b.get("retrieval_at_1"),
        "verdict": (
            _classify_diff(d.get("retrieval_at_1"), b.get("retrieval_at_1"), higher_is_better=True, eps_abs=0.01, eps_rel=0.05)[0]
            if has_b else _classify_absolute(d.get("retrieval_at_1"), strong_threshold=0.25, weak_threshold=0.15)
        ),
    })

    return "\n".join(lines) + "\n", findings


def build_av_grounding_section(
    d: dict[str, Any], b: dict[str, Any] | None,
) -> tuple[str, list[dict]]:
    has_b = b is not None
    b = b or {}
    findings: list[dict] = []
    lines: list[str] = ["## AV grounding", ""]

    judge_cur = d.get("judge") or {}
    judge_base = (b.get("judge") or {}) if has_b else {}
    if not judge_cur.get("by_variant") and not d.get("av_diversity", {}).get("by_ptype"):
        lines.append("- _missing_: no `llm_judge.jsonl` and no `av_samples.jsonl`.")
        return "\n".join(lines) + "\n", findings

    if judge_cur.get("by_variant"):
        lines.append(f"LLM judge: n={judge_cur.get('n', 0)} rows across variants "
                     f"{sorted(judge_cur['by_variant'].keys())}.")
        lines.append("")
        lines.append("| Variant | Axis | Pass rate (cur) | Pass rate (base) | Δ | Verdict |")
        lines.append("|---|---|---:|---:|---:|---|")
        for variant in sorted(judge_cur["by_variant"]):
            cur_axes = judge_cur["by_variant"][variant]
            base_axes = (judge_base.get("by_variant") or {}).get(variant, {})
            for axis_label, axis_key in (
                ("B (grounding)", "grounding"),
                ("C (template_distinguishable)", "template_distinguishable"),
                ("A (appropriateness)", "appropriateness"),
            ):
                cur_v = cur_axes.get(axis_key)
                base_v = base_axes.get(axis_key) if has_b else None
                if has_b:
                    v_label, delta_s = _classify_diff(cur_v, base_v, higher_is_better=True, eps_abs=0.02, eps_rel=0.05)
                else:
                    strong = 0.55 if axis_key != "appropriateness" else 0.80
                    weak = 0.40 if axis_key != "appropriateness" else 0.60
                    v_label = _classify_absolute(cur_v, strong_threshold=strong, weak_threshold=weak)
                    delta_s = "—"
                lines.append(f"| `{variant}` | {axis_label} | {_fmt_pct(cur_v)} | {_fmt_pct(base_v)} | {delta_s} | {v_label} |")
                findings.append({
                    "axis": "av_grounding", "metric": f"judge_{axis_key}_{variant}",
                    "value": cur_v, "baseline": base_v, "verdict": v_label,
                })
    else:
        lines.append("- _missing_: no `llm_judge.jsonl` parsed.")

    av = d.get("av_diversity") or {}
    av_b = (b.get("av_diversity") or {}) if has_b else {}
    if av.get("by_ptype"):
        lines.append("")
        lines.append(f"AV samples diversity: n_rows={av.get('n_rows', 0)}.")
        lines.append("")
        lines.append("| Position | n | distinct-first / n | TF cosine | CL cosine | CL−TF cosine drop |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for pt in sorted(av["by_ptype"]):
            row = av["by_ptype"][pt]
            base_row = (av_b.get("by_ptype") or {}).get(pt, {}) if has_b else {}
            ratio = row.get("distinct_first_ratio")
            base_ratio = base_row.get("distinct_first_ratio") if has_b else None
            if has_b:
                v_label, _ = _classify_diff(ratio, base_ratio, higher_is_better=True, eps_abs=0.05, eps_rel=0.10)
            else:
                v_label = _classify_absolute(ratio, strong_threshold=0.5, weak_threshold=0.25)
            findings.append({
                "axis": "av_grounding", "metric": f"av_distinct_first_ratio_{pt}",
                "value": ratio, "baseline": base_ratio, "verdict": v_label,
            })
            lines.append(
                f"| `{pt}` | {row['n']} | "
                f"{_fmt_pct(ratio)} | "
                f"{_fmt_num(row.get('tf_cosine_mean'))} | "
                f"{_fmt_num(row.get('cl_cosine_mean'))} | "
                f"{_fmt_num(row.get('cl_cosine_drop_vs_tf'))} |"
            )

    return "\n".join(lines) + "\n", findings


def build_behavior_section(
    d: dict[str, Any], b: dict[str, Any] | None,
) -> tuple[str, list[dict]]:
    has_b = b is not None
    b = b or {}
    findings: list[dict] = []
    lines: list[str] = ["## Behavioral signal", ""]

    if (
        d.get("sim_correct_success") is None
        and d.get("ar_mse_last_train") is None
        and d.get("ce_last_train") is None
    ):
        lines.append("- _missing_: no `sim_ab.json` numeric success rates and no `metrics.jsonl` train tail.")
        return "\n".join(lines) + "\n", findings

    lines.append(_line_for(
        "Sim A/B: correct-success rate",
        d.get("sim_correct_success"), b.get("sim_correct_success"),
        fmt="pct", higher_is_better=True, strong=0.30, weak=0.15,
        eps_abs=0.02, eps_rel=0.10,
        has_baseline=has_b,
    ))
    lines.append(_line_for(
        "Sim A/B: wrong-success rate",
        d.get("sim_wrong_success"), b.get("sim_wrong_success"),
        fmt="pct", higher_is_better=False, strong=0.05, weak=0.20,
        eps_abs=0.02, eps_rel=0.10,
        has_baseline=has_b,
        extra="should be ≤ correct; if not, AV isn't doing causal work",
    ))
    lines.append(_line_for(
        "Sim A/B: correct − wrong",
        d.get("sim_correct_minus_wrong"), b.get("sim_correct_minus_wrong"),
        higher_is_better=True, strong=0.10, weak=0.0,
        eps_abs=0.02, eps_rel=0.10,
        has_baseline=has_b,
    ))
    lines.append(_line_for(
        "Sim A/B: correct − baseline",
        d.get("sim_correct_minus_baseline"), b.get("sim_correct_minus_baseline"),
        higher_is_better=True, strong=0.05, weak=-0.10,
        eps_abs=0.02, eps_rel=0.10,
        has_baseline=has_b,
    ))
    lines.append(_line_for(
        "AR MSE (last train row)",
        d.get("ar_mse_last_train"), b.get("ar_mse_last_train"),
        higher_is_better=False, strong=0.05, weak=0.5,
        eps_abs=0.005, eps_rel=0.10,
        has_baseline=has_b,
        extra="action consistency proxy — closer to zero = action head agrees with steered activations",
    ))
    lines.append(_line_for(
        "Total loss (last train row)",
        d.get("loss_last_train"), b.get("loss_last_train"),
        higher_is_better=False, strong=1.5, weak=3.0,
        eps_abs=0.05, eps_rel=0.05,
        has_baseline=has_b,
    ))
    lines.append(_line_for(
        "CE (last train row)",
        d.get("ce_last_train"), b.get("ce_last_train"),
        higher_is_better=False, strong=0.7, weak=1.5,
        eps_abs=0.05, eps_rel=0.05,
        has_baseline=has_b,
    ))

    findings.append({
        "axis": "behavior", "metric": "sim_correct_success",
        "value": d.get("sim_correct_success"), "baseline": b.get("sim_correct_success"),
        "verdict": (
            _classify_diff(d.get("sim_correct_success"), b.get("sim_correct_success"), higher_is_better=True, eps_abs=0.02, eps_rel=0.10)[0]
            if has_b else _classify_absolute(d.get("sim_correct_success"), strong_threshold=0.30, weak_threshold=0.15)
        ),
    })
    return "\n".join(lines) + "\n", findings


def build_position_breakdown_section(
    d: dict[str, Any], b: dict[str, Any] | None,
) -> tuple[str, list[dict]]:
    has_b = b is not None
    b = b or {}
    findings: list[dict] = []
    lines: list[str] = ["## Position-type breakdown (which positions are weakest)", ""]

    pt_metrics: dict[str, dict[str, float | None]] = {}

    for pt, blob in (d.get("retrieval_by_position") or {}).items():
        if not isinstance(blob, dict):
            continue
        pt_metrics.setdefault(pt, {})
        pt_metrics[pt]["retrieval_margin"] = blob.get("margin")
        pt_metrics[pt]["retrieval_at_1"] = blob.get("retrieval_at_1")
        pt_metrics[pt]["retrieval_at_5"] = blob.get("retrieval_at_5")

    for pt, v in (d.get("fve_by_ptype") or {}).items():
        pt_metrics.setdefault(pt, {})
        pt_metrics[pt]["fve"] = v
    for pt, v in (d.get("closed_greedy_cosine_by_ptype") or {}).items():
        pt_metrics.setdefault(pt, {})
        pt_metrics[pt]["closed_greedy_cosine"] = v
    for pt, blob in (d.get("hard_neg_by_ptype") or {}).items():
        if not isinstance(blob, dict):
            continue
        pt_metrics.setdefault(pt, {})
        pt_metrics[pt]["hard_neg_median_cos_top1"] = blob.get("median_cos_top1")
    for pt, blob in (d.get("episode_cosine_gap_by_ptype") or {}).items():
        if not isinstance(blob, dict):
            continue
        pt_metrics.setdefault(pt, {})
        pt_metrics[pt]["episode_cos_gap"] = blob.get("gap")
    for pt, blob in (d.get("suite_probe_by_ptype") or {}).items():
        if not isinstance(blob, dict):
            continue
        pt_metrics.setdefault(pt, {})
        pt_metrics[pt]["suite_probe_acc"] = blob.get("accuracy")

    if not pt_metrics:
        lines.append("- _missing_: no per-position data anywhere.")
        return "\n".join(lines) + "\n", findings

    positions = sorted(pt_metrics.keys())
    headers = [
        "retrieval_margin", "retrieval_at_1", "retrieval_at_5",
        "fve", "closed_greedy_cosine",
        "hard_neg_median_cos_top1", "episode_cos_gap", "suite_probe_acc",
    ]
    lines.append("| Position | " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * (len(headers) + 1)) + "|")
    for pt in positions:
        row = pt_metrics[pt]
        cells = [_fmt_num(row.get(h)) for h in headers]
        lines.append(f"| `{pt}` | " + " | ".join(cells) + " |")

    weak_signals: dict[str, list[str]] = {}
    for pt in positions:
        weaknesses: list[str] = []
        rm = pt_metrics[pt].get("retrieval_margin")
        if isinstance(rm, (int, float)) and rm < 0.02:
            weaknesses.append(f"retrieval_margin={rm:.3f}<0.02")
        r1 = pt_metrics[pt].get("retrieval_at_1")
        if isinstance(r1, (int, float)) and r1 < 0.15:
            weaknesses.append(f"R@1={r1:.3f}<0.15")
        fve = pt_metrics[pt].get("fve")
        if isinstance(fve, (int, float)) and fve < 0.0:
            weaknesses.append(f"FVE={fve:.3f}<0 (worse than mean-predictor)")
        hn = pt_metrics[pt].get("hard_neg_median_cos_top1")
        if isinstance(hn, (int, float)) and hn > 0.97:
            weaknesses.append(f"hard_neg_median={hn:.3f}>0.97 (input collapse hypothesis)")
        gap = pt_metrics[pt].get("episode_cos_gap")
        if isinstance(gap, (int, float)) and gap < 0.01:
            weaknesses.append(f"episode_cos_gap={gap:.4f} (raw h barely scene-specific)")
        if weaknesses:
            weak_signals[pt] = weaknesses

    if weak_signals:
        lines.append("")
        lines.append("**Weakest positions:**")
        for pt, ws in weak_signals.items():
            lines.append(f"- `{pt}`: " + "; ".join(ws))
            findings.append({
                "axis": "position", "metric": f"position_weakness_{pt}",
                "value": ws, "verdict": "WEAK",
            })
    else:
        lines.append("")
        lines.append("No position triggered a per-position weakness flag.")

    return "\n".join(lines) + "\n", findings


# ---------------------------------------------------------------------------
# Action list
# ---------------------------------------------------------------------------


def build_action_list(
    d: dict[str, Any], findings: list[dict],
) -> str:
    """Translate weak / regressed findings into concrete flag-level suggestions.

    Ranked by an ad-hoc impact heuristic: behavior > grounding > retrieval >
    reconstruction > position-detail.
    """
    actions: list[tuple[int, str]] = []

    rm = d.get("retrieval_margin")
    if isinstance(rm, (int, float)) and rm < 0.05:
        actions.append((
            70,
            f"Retrieval margin is {rm:.3f} (< 0.05). To improve discrimination: "
            "raise `--ar-contrastive-weight` (e.g. 0.5→1.0) and/or lower "
            "`--ar-nce-temperature` (0.1→0.07) in `scripts/training/run_sft.py`. "
            "If hard-neg median cos_top1 > 0.97 (see extraction_diag), "
            "regenerate harder negatives before re-training.",
        ))

    nce = d.get("ar_nce_last_train")
    if isinstance(nce, (int, float)) and nce > 3.0:
        actions.append((
            65,
            f"AR-NCE loss did not collapse (last train row = {nce:.3f}). "
            "Re-check `--ar-contrastive-weight` is > 0 in `run_sft.py`; if 0, NCE "
            "is logging-only. If > 0, lower `--ar-nce-temperature` or raise the weight.",
        ))

    hn = d.get("hard_neg_by_ptype") or {}
    weak_ptypes = [pt for pt, blob in hn.items()
                   if isinstance(blob, dict) and isinstance(blob.get("median_cos_top1"), (int, float))
                   and blob["median_cos_top1"] > 0.97]
    if weak_ptypes:
        actions.append((
            72,
            f"Hard-negative median cos_top1 > 0.97 for {weak_ptypes}. "
            "This is the 'input-side collapse' verdict from extraction_diag — "
            "before tuning training, re-run the layer/strategy sweep "
            "(`scripts/eval/probe_extraction_sweep.py`) and switch "
            "`--activations-root` to the winning combination before the next `run_sft.py`.",
        ))

    judge = (d.get("judge") or {}).get("by_variant") or {}
    pred_judge = judge.get("av_pred") or judge.get("gold") or {}
    grd = pred_judge.get("grounding") if isinstance(pred_judge, dict) else None
    tdis = pred_judge.get("template_distinguishable") if isinstance(pred_judge, dict) else None
    if isinstance(grd, (int, float)) and grd < 0.55:
        actions.append((
            80,
            f"Judge axis B (grounding) pass rate = {grd:.2%} (< 55%). "
            "AV captions hallucinate scene content. Try: "
            "(a) `--judge-reward-weight 0.5` in `scripts/training/run_grpo.py` to RL "
            "against the same judge, (b) raise `--av-weight` in `run_sft.py`, "
            "(c) re-mine labels with a stricter prompt before re-running SFT.",
        ))
    if isinstance(tdis, (int, float)) and tdis < 0.50:
        actions.append((
            78,
            f"Judge axis C (template_distinguishable) pass rate = {tdis:.2%} (< 50%). "
            "Captions look generic/templated. Try `--ar-contrastive-weight 1.0` "
            "and `--balance-position-mix` in `run_sft.py`; consider adding "
            "anti-template counterfactuals via `--sim-counterfactual-pairs-path` "
            "in `run_grpo.py`.",
        ))

    av = (d.get("av_diversity") or {}).get("by_ptype") or {}
    low_div = [pt for pt, row in av.items()
               if isinstance(row.get("distinct_first_ratio"), (int, float))
               and row["distinct_first_ratio"] < 0.25]
    if low_div:
        actions.append((
            55,
            f"Generated AV first-line diversity is < 25% for {low_div}. "
            "This is template collapse. Train longer with "
            "`--ar-contrastive-weight > 0`, raise `--rollout-temperature` (e.g. 1.2) "
            "in `run_grpo.py`, and verify `--min-bullets` in `run_sft.py` is set "
            "(or remove the cap if currently small).",
        ))

    fve = d.get("fve_total")
    if isinstance(fve, (int, float)) and fve < 0.0:
        actions.append((
            60,
            f"Total FVE = {fve:.3f} (< 0, worse than predicting the mean). "
            "Reconstruction head is under-fitting. Increase `--total-steps` and "
            "`--ar-weight` in `run_sft.py`; if `--ar-clip-target-scaled` is set "
            "very low, raise it. Re-check `--alpha`.",
        ))

    cl_gap = d.get("closed_loop_cosine_gap")
    if isinstance(cl_gap, (int, float)) and cl_gap > 0.10:
        actions.append((
            50,
            f"Teacher-forced − closed-loop cosine gap = {cl_gap:+.3f} (> 0.10). "
            "Sampling drifts off-manifold. In `run_sft.py`, enable "
            "`--eval-closed-loop` plus a richer `--closed-loop-temps 0.0 0.7 1.0` "
            "so this is monitored; lower the decoding temperature for inference; "
            "or train with mild scheduled sampling.",
        ))

    sim_cs = d.get("sim_correct_success")
    sim_cmw = d.get("sim_correct_minus_wrong")
    if isinstance(sim_cs, (int, float)) and sim_cs <= 0.0:
        actions.append((
            95,
            f"Sim A/B correct-success rate = {sim_cs:.2%}. The steered model isn't "
            "actually solving tasks. First, verify the sim server is up (sim_ab.json "
            "shows server_returncode != 0 ⇒ server crashed). Then, if non-zero is "
            "achievable, set `--sim-reward-weight 0.5` and "
            "`--sim-counterfactual-pairs-path <pairs.jsonl>` in `run_grpo.py` to "
            "co-train with behavioural reward.",
        ))
    elif isinstance(sim_cmw, (int, float)) and sim_cmw < 0.05:
        actions.append((
            85,
            f"Sim A/B correct−wrong = {sim_cmw:+.3f}. AV isn't differentiating "
            "behaviour. Increase `--sim-reward-weight` in `run_grpo.py` and raise "
            "`--rollouts-per-activation` (e.g. 4→8) for lower-variance advantages.",
        ))

    for f in findings:
        if f.get("verdict") == "REGRESSED" and f.get("metric"):
            actions.append((
                90,
                f"Regression on `{f['metric']}` (current={_fmt_num(f.get('value'))}, "
                f"baseline={_fmt_num(f.get('baseline'))}). Diff the two run configs "
                "(activations-root, alpha, batch size, ar-weight, learning rate) and "
                "consider rolling back the offending knob before adding new ones.",
            ))

    if not actions:
        return "## Action list (ranked by impact)\n\nNo weak or regressed signals tripped a recommendation. Re-run with `--baseline-dir` to surface deltas.\n"

    actions.sort(key=lambda x: x[0], reverse=True)
    seen: set[str] = set()
    deduped: list[str] = []
    for _, msg in actions:
        if msg in seen:
            continue
        seen.add(msg)
        deduped.append(msg)
    body = ["## Action list (ranked by impact)", ""]
    for i, msg in enumerate(deduped, 1):
        body.append(f"{i}. {msg}")
    body.append("")
    return "\n".join(body) + "\n"


# ---------------------------------------------------------------------------
# Top-level render
# ---------------------------------------------------------------------------


def _section_status_counter(findings: list[dict]) -> str:
    c = Counter(f.get("verdict") for f in findings)
    parts = []
    for k in ("IMPROVED", "STABLE", "REGRESSED", "UNCHECKED", "STRONG", "WEAK", "UNKNOWN"):
        if c.get(k):
            parts.append(f"{k}={c[k]}")
    return ", ".join(parts) if parts else "(no findings)"


def render_report(
    sft_dir: Path,
    baseline_dir: Path | None,
    bundle: dict[str, Any],
    baseline_bundle: dict[str, Any] | None,
) -> str:
    d = derive(bundle)
    b = derive(baseline_bundle) if baseline_bundle else None
    has_b = b is not None

    head = [
        "# SFT improvements report",
        "",
        f"- SFT dir: `{sft_dir}`",
        f"- Baseline dir: `{baseline_dir}`" if has_b else "- Baseline dir: _(none — absolute mode)_",
        f"- Overall scorecard verdict: **{d.get('overall_verdict') or '—'}**"
        + (f" (baseline: {b.get('overall_verdict') or '—'})" if has_b else ""),
        f"- Training step recorded: {d.get('training_step') or '—'}",
        "",
    ]
    if bundle["_missing"]:
        head.append("**Missing eval files (treated as soft-skips):**")
        for m in bundle["_missing"]:
            head.append(f"- `{m}`")
        head.append("")
    if has_b and baseline_bundle and baseline_bundle.get("_missing"):
        head.append("**Missing baseline eval files (treated as soft-skips):**")
        for m in baseline_bundle["_missing"]:
            head.append(f"- `{m}` (baseline)")
        head.append("")
    if has_b:
        head.append("> Diff mode. Labels: **IMPROVED / STABLE / REGRESSED / UNCHECKED**. "
                    "STABLE band = max(0.5% absolute, 5% relative) per metric.")
    else:
        head.append("> Absolute mode (no baseline). Labels: **STRONG / WEAK / UNKNOWN**. "
                    "These are dataset-agnostic heuristic thresholds; calibrate against your own runs.")
    head.append("")

    all_findings: list[dict] = []
    sections: list[str] = []
    for builder in (
        build_reconstruction_section,
        build_retrieval_section,
        build_av_grounding_section,
        build_behavior_section,
        build_position_breakdown_section,
    ):
        body, findings = builder(d, b)
        all_findings.extend(findings)
        sections.append(body)

    summary = ["## TL;DR", "", f"- Findings rolled up: {_section_status_counter(all_findings)}"]
    if has_b:
        improved = [f["metric"] for f in all_findings if f.get("verdict") == "IMPROVED"]
        regressed = [f["metric"] for f in all_findings if f.get("verdict") == "REGRESSED"]
        if improved:
            summary.append(f"- Top improvements: {', '.join(improved[:5])}")
        if regressed:
            summary.append(f"- Top regressions (read first): {', '.join(regressed[:5])}")
    else:
        weak = [f["metric"] for f in all_findings if f.get("verdict") == "WEAK"]
        if weak:
            summary.append(f"- Weak signals to address: {', '.join(weak[:5])}")
    summary.append("")

    action_md = build_action_list(d, all_findings)

    return (
        "\n".join(head)
        + "\n".join(summary)
        + "\n"
        + "\n".join(sections)
        + "\n"
        + action_md
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sft-dir", required=True,
                   help="SFT checkpoint dir whose eval outputs we summarise.")
    p.add_argument("--baseline-dir", default=None,
                   help="Optional baseline checkpoint dir; produces a diff report.")
    p.add_argument("--output", default=None,
                   help="Output markdown path. Defaults to <sft-dir>/improvements.md.")
    p.add_argument("--log-level", default="WARNING")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    sft_dir = Path(args.sft_dir)
    if not sft_dir.exists():
        print(f"FAIL: sft-dir not found: {sft_dir}")
        return 1
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else None
    if baseline_dir is not None and not baseline_dir.exists():
        print(f"FAIL: baseline-dir not found: {baseline_dir}")
        return 1

    bundle = load_bundle(sft_dir)
    baseline_bundle = load_bundle(baseline_dir) if baseline_dir else None

    md = render_report(sft_dir, baseline_dir, bundle, baseline_bundle)

    out_path = Path(args.output) if args.output else (sft_dir / "improvements.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)

    n_missing_cur = len(bundle["_missing"])
    n_missing_base = len(baseline_bundle["_missing"]) if baseline_bundle else 0
    mode = "diff" if baseline_bundle else "absolute"
    print(f"OK  wrote {out_path}  mode={mode}  missing_cur={n_missing_cur}  missing_base={n_missing_base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
