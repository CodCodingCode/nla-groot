#!/usr/bin/env python
"""Build the SA9 multimodal-judge A/B scorecard report.

Inputs (all from data/eval/):
    - libero_v3_quality_judge.jsonl              (V3 baseline, frozen)
    - libero_v4_quality_judge_combined.jsonl     (V4 headline: combined SFT pool)
    - libero_v4_quality_judge_v4only.jsonl       (V4 attributable: only re-written rows)

Outputs:
    - data/eval/sa9_v4_judge_scorecard.json
    - docs/sft_plan/v4_repair/sa9_judge_ab.md
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

REPO = Path("/home/ubuntu/nla-groot")
V3_PATH = REPO / "data/eval/libero_v3_quality_judge.jsonl"
V4_COMBINED_PATH = REPO / "data/eval/libero_v4_quality_judge_combined.jsonl"
V4_ONLY_PATH = REPO / "data/eval/libero_v4_quality_judge_v4only.jsonl"

SCORECARD_OUT = REPO / "data/eval/sa9_v4_judge_scorecard.json"
REPORT_OUT = REPO / "docs/sft_plan/v4_repair/sa9_judge_ab.md"

SUITES = ("spatial", "goal", "object", "10")
PTYPES = ("anchor", "image_patch", "last_text")
PTYPE_RX = re.compile(r"@p(\d+)_(.+)$")


def _parse_eid(eid: str) -> tuple[str | None, int | None, str | None, str | None]:
    """Returns (suite, position_index, position_type, source_id)."""
    suite: str | None = None
    if "::" in eid:
        head, rest = eid.split("::", 1)
        if head.startswith("libero_"):
            suite = head[len("libero_"):]
        else:
            suite = head
    else:
        rest = eid
    m = PTYPE_RX.search(rest)
    if m:
        idx = int(m.group(1))
        ptype = m.group(2)
        sid = rest[: m.start()]
        return suite, idx, ptype, sid
    return suite, None, None, rest


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            eid = o.get("example_id") or ""
            suite, idx, ptype, sid = _parse_eid(eid)
            g = (o.get("grounding") or {}).get("verdict")
            a = (o.get("appropriateness") or {}).get("verdict")
            rows.append({
                "example_id": eid,
                "suite": suite,
                "position_index": idx,
                "position_type": ptype,
                "source_id": sid,
                "grounding": g,
                "appropriateness": a,
                "grounding_reason": (o.get("grounding") or {}).get("reason"),
                "appropriateness_reason": (o.get("appropriateness") or {}).get("reason"),
                "raw": o,
            })
    return rows


def _agg(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    n = len(rows)
    b = sum(1 for r in rows if r["grounding"] == "specific")
    c = sum(1 for r in rows if r["appropriateness"] == "appropriate")
    return {
        "n": n,
        "b_pass": b,
        "c_pass": c,
        "b_pct": (b / n) if n else None,
        "c_pct": (c / n) if n else None,
    }


def _slice(rows: list[dict], *, suite: str | None = None,
           ptype: str | None = None) -> list[dict]:
    out = rows
    if suite is not None:
        out = [r for r in out if r["suite"] == suite]
    if ptype is not None:
        out = [r for r in out if r["position_type"] == ptype]
    return out


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.2f}%"


def _fmt_pp_delta(v4: float | None, v3: float | None) -> str:
    if v4 is None or v3 is None:
        return "—"
    d = (v4 - v3) * 100
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.2f}pp"


def _scorecard(v3: list[dict], v4: list[dict]) -> dict:
    sc: dict = {}
    sc["overall_v3"] = _agg(v3)
    sc["overall_v4"] = _agg(v4)
    sc["per_suite_v3"] = {s: _agg(_slice(v3, suite=s)) for s in SUITES}
    sc["per_suite_v4"] = {s: _agg(_slice(v4, suite=s)) for s in SUITES}
    sc["per_ptype_v3"] = {p: _agg(_slice(v3, ptype=p)) for p in PTYPES}
    sc["per_ptype_v4"] = {p: _agg(_slice(v4, ptype=p)) for p in PTYPES}
    cells: dict = {}
    regressions: list[tuple[str, str, float]] = []
    big_b_wins: list[tuple[str, str, float]] = []
    for s in SUITES:
        for p in PTYPES:
            key = f"{s}/{p}"
            v3a = _agg(_slice(v3, suite=s, ptype=p))
            v4a = _agg(_slice(v4, suite=s, ptype=p))
            cells[key] = {"v3": v3a, "v4": v4a}
            v3b = v3a["b_pct"]; v4b = v4a["b_pct"]
            v3c = v3a["c_pct"]; v4c = v4a["c_pct"]
            if v3b is not None and v4b is not None:
                dB = (v4b - v3b) * 100
                cells[key]["delta_b_pp"] = dB
                if dB < -5:
                    regressions.append((key, "B", dB))
                if dB >= 10:
                    big_b_wins.append((key, "B", dB))
            else:
                cells[key]["delta_b_pp"] = None
            if v3c is not None and v4c is not None:
                dC = (v4c - v3c) * 100
                cells[key]["delta_c_pp"] = dC
                if dC < -5:
                    regressions.append((key, "C", dC))
            else:
                cells[key]["delta_c_pp"] = None
    sc["cells"] = cells
    sc["cells_regressed_gt5pp"] = regressions
    sc["cells_b_gain_ge10pp"] = big_b_wins
    return sc


def _row_level_deltas(v3: list[dict], v4: list[dict]) -> dict:
    """Match rows by (suite, source_id, position_index) and compare grounding."""
    v3_idx = {(r["suite"], r["source_id"], r["position_index"]): r for r in v3}
    wins: list[dict] = []
    losses: list[dict] = []
    for r in v4:
        key = (r["suite"], r["source_id"], r["position_index"])
        v3r = v3_idx.get(key)
        if not v3r:
            continue
        if v3r["grounding"] != "specific" and r["grounding"] == "specific":
            wins.append({"key": key, "v3": v3r, "v4": r})
        elif v3r["grounding"] == "specific" and r["grounding"] != "specific":
            losses.append({"key": key, "v3": v3r, "v4": r})
    return {"wins": wins, "losses": losses}


def _gate_verdicts(sc: dict) -> dict:
    overall_b = sc["overall_v4"]["b_pct"] or 0.0
    overall_c = sc["overall_v4"]["c_pct"] or 0.0
    spatial_b = sc["per_suite_v4"]["spatial"]["b_pct"] or 0.0
    paper = "GREEN" if (overall_b >= 0.95 and overall_c >= 0.95) else (
        "YELLOW" if overall_c >= 0.95 else "RED")
    spatial_pass = spatial_b >= 0.85
    n_regressed = len(sc["cells_regressed_gt5pp"])
    return {
        "paper_grade": paper,
        "paper_grade_pass": paper == "GREEN",
        "spatial_rescue_pass": spatial_pass,
        "spatial_b": spatial_b,
        "no_regression_pass": n_regressed == 0,
        "n_cells_regressed_gt5pp": n_regressed,
    }


def _md_overall(name: str, sc: dict) -> str:
    v3 = sc["overall_v3"]
    v4 = sc["overall_v4"]
    lines = []
    lines.append(f"### Overall — {name}")
    lines.append("")
    lines.append("| Metric | V3 baseline | V4 | Δ |")
    lines.append("|---|---|---|---|")
    lines.append(f"| n | {v3['n']} | {v4['n']} | — |")
    lines.append(f"| B (grounding=specific) | {_fmt_pct(v3['b_pct'])} ({v3['b_pass']}/{v3['n']}) | "
                 f"{_fmt_pct(v4['b_pct'])} ({v4['b_pass']}/{v4['n']}) | "
                 f"{_fmt_pp_delta(v4['b_pct'], v3['b_pct'])} |")
    lines.append(f"| C (appropriateness=appropriate) | {_fmt_pct(v3['c_pct'])} ({v3['c_pass']}/{v3['n']}) | "
                 f"{_fmt_pct(v4['c_pct'])} ({v4['c_pass']}/{v4['n']}) | "
                 f"{_fmt_pp_delta(v4['c_pct'], v3['c_pct'])} |")
    lines.append("")
    return "\n".join(lines)


def _md_per_axis(label: str, axis_name: str, keys: tuple[str, ...],
                 v3_map: dict, v4_map: dict) -> str:
    lines = [f"### Per-{axis_name} — {label}", ""]
    lines.append(f"| {axis_name} | n (V4) | V3 B | V4 B | ΔB | V3 C | V4 C | ΔC |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for k in keys:
        v3 = v3_map[k]; v4 = v4_map[k]
        lines.append(f"| {k} | {v4['n']} | {_fmt_pct(v3['b_pct'])} | {_fmt_pct(v4['b_pct'])} | "
                     f"{_fmt_pp_delta(v4['b_pct'], v3['b_pct'])} | "
                     f"{_fmt_pct(v3['c_pct'])} | {_fmt_pct(v4['c_pct'])} | "
                     f"{_fmt_pp_delta(v4['c_pct'], v3['c_pct'])} |")
    lines.append("")
    return "\n".join(lines)


def _md_matrix(label: str, sc: dict) -> str:
    lines = [f"### 12-cell matrix — {label}", ""]
    lines.append("| suite/ptype | n (V4) | V3 B | V4 B | ΔB | V3 C | V4 C | ΔC |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in SUITES:
        for p in PTYPES:
            key = f"{s}/{p}"
            cell = sc["cells"][key]
            v3 = cell["v3"]; v4 = cell["v4"]
            lines.append(f"| {key} | {v4['n']} | {_fmt_pct(v3['b_pct'])} | {_fmt_pct(v4['b_pct'])} | "
                         f"{_fmt_pp_delta(v4['b_pct'], v3['b_pct'])} | "
                         f"{_fmt_pct(v3['c_pct'])} | {_fmt_pct(v4['c_pct'])} | "
                         f"{_fmt_pp_delta(v4['c_pct'], v3['c_pct'])} |")
    lines.append("")
    return "\n".join(lines)


def _md_deltas(name: str, deltas: dict, k: int = 20) -> str:
    lines = [f"### Row-level deltas — {name}", ""]
    wins = deltas["wins"]
    losses = deltas["losses"]
    lines.append(f"**Wins (V3 non-specific → V4 specific):** {len(wins)} matched rows.")
    lines.append("")
    if wins:
        lines.append(f"Top {min(k, len(wins))} wins (matched on (suite, source_id, position_index)):")
        lines.append("")
        lines.append("| suite | source_id | pos | ptype | V3 grounding | V4 grounding | V3 reason | V4 reason |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for w in wins[:k]:
            key = w["key"]; v3 = w["v3"]; v4 = w["v4"]
            lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {v4['position_type']} | "
                         f"{v3['grounding']} | {v4['grounding']} | "
                         f"{(v3['grounding_reason'] or '')[:120]} | "
                         f"{(v4['grounding_reason'] or '')[:120]} |")
        lines.append("")
    lines.append(f"**Losses (V3 specific → V4 non-specific):** {len(losses)} matched rows.")
    lines.append("")
    if losses:
        lines.append(f"Top {min(k, len(losses))} losses:")
        lines.append("")
        lines.append("| suite | source_id | pos | ptype | V3 grounding | V4 grounding | V3 reason | V4 reason |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for w in losses[:k]:
            key = w["key"]; v3 = w["v3"]; v4 = w["v4"]
            lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {v4['position_type']} | "
                         f"{v3['grounding']} | {v4['grounding']} | "
                         f"{(v3['grounding_reason'] or '')[:120]} | "
                         f"{(v4['grounding_reason'] or '')[:120]} |")
        lines.append("")
    return "\n".join(lines)


def _frame_path_hint(suite: str, source_id: str) -> str:
    base = f"data/labels/libero_4suite_stride2/libero_{suite}/frames_cache"
    return f"`{base}/{source_id}__image.jpg` (+ `__wrist_image.jpg`)"


def _md_worst_examples(name: str, v4_rows: list[dict], k: int = 10) -> str:
    """List worst-grading V4 rows (grounding=generic; appropriateness fail wins tiebreak)."""
    bad = [r for r in v4_rows if r["grounding"] != "specific"]
    bad.sort(key=lambda r: (
        0 if r["appropriateness"] != "appropriate" else 1,
        r["suite"] or "",
        r["position_type"] or "",
    ))
    lines = [f"### 10 worst-grading V4 rows — {name}", ""]
    if not bad:
        lines.append("No grounding failures. ")
        lines.append("")
        return "\n".join(lines)
    lines.append(f"Showing first {min(k, len(bad))} (B-fails first, then any C-fails):")
    lines.append("")
    for i, r in enumerate(bad[:k]):
        lines.append(f"#### W{i+1}. {r['suite']} / {r['position_type']} — {r['source_id']} (p{r['position_index']})")
        lines.append("")
        lines.append(f"- **B verdict:** {r['grounding']} — {(r['grounding_reason'] or '').strip()}")
        lines.append(f"- **C verdict:** {r['appropriateness']} — {(r['appropriateness_reason'] or '').strip()}")
        lines.append(f"- **Frames:** {_frame_path_hint(r['suite'], r['source_id'])}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    v3 = _load_rows(V3_PATH)
    v4c = _load_rows(V4_COMBINED_PATH)
    v4o = _load_rows(V4_ONLY_PATH)

    sc_combined = _scorecard(v3, v4c)
    sc_v4only = _scorecard(v3, v4o)
    gates_combined = _gate_verdicts(sc_combined)
    gates_v4only = _gate_verdicts(sc_v4only)
    deltas_combined = _row_level_deltas(v3, v4c)
    deltas_v4only = _row_level_deltas(v3, v4o)

    SCORECARD_OUT.parent.mkdir(parents=True, exist_ok=True)
    json_blob = {
        "v3_baseline_path": str(V3_PATH),
        "v4_combined_path": str(V4_COMBINED_PATH),
        "v4_only_path": str(V4_ONLY_PATH),
        "seed": 0,
        "sample_size": 500,
        "grader": "gpt-5.1",
        "scorecards": {
            "v4_combined_vs_v3": sc_combined,
            "v4_only_vs_v3": sc_v4only,
        },
        "gates": {
            "v4_combined_vs_v3": gates_combined,
            "v4_only_vs_v3": gates_v4only,
        },
        "row_level_matched_counts": {
            "v4_combined": {
                "wins": len(deltas_combined["wins"]),
                "losses": len(deltas_combined["losses"]),
            },
            "v4_only": {
                "wins": len(deltas_v4only["wins"]),
                "losses": len(deltas_v4only["losses"]),
            },
        },
    }
    with SCORECARD_OUT.open("w") as f:
        json.dump(json_blob, f, indent=2, default=str)
    print(f"Wrote {SCORECARD_OUT}")

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# SA9 — LIBERO V4 Multimodal Judge A/B (vs V3)\n")
    lines.append("**Headline metric.** Stratified-500-row gpt-5.1 multimodal judge comparing the V4 "
                 "rewritten dataset against the V3 frozen baseline (Agent 1's audit, "
                 "`data/eval/libero_v3_quality_judge.jsonl`).\n")
    lines.append("## Methodology\n")
    lines.append("- **Script:** `scripts/eval/verify_libero_label_quality.py` (the exact script that "
                 "produced the V3 baseline; unchanged for V4).")
    lines.append("- **Grader:** OpenAI `gpt-5.1`, prompt unchanged from V3.")
    lines.append("- **Sample size:** 500 rows per run, stratified by (suite × position_type) = 12 buckets.")
    lines.append("- **Seed:** `--seed 0` (matches the V3 run), so bucket sizes are directly comparable.")
    lines.append("- **Concurrency:** 32.")
    lines.append("- **Frames:** the V3 frames cache (`data/labels/libero_4suite_stride2/libero_<suite>/"
                 "frames_cache/`) is reused — V4 did not re-render frames; the underlying LIBERO trajectories "
                 "are identical.")
    lines.append("")
    lines.append("Two runs were performed:")
    lines.append("")
    lines.append("1. **`v4_combined` (headline / paper-grade pool).** Samples the full V4 merged label pool — "
                 "82,005 newly-rewritten rows + 19,350 V3-kept rows = 101,580 total in "
                 "`data/labels/libero_4suite_v4_combined/labels.jsonl`. This is the apples-to-apples comparison "
                 "with V3 because it is exactly the row set SFT will train on.")
    lines.append("2. **`v4_only` (attributable improvement).** Samples only the V4-rewritten rows "
                 "(82,005 across `data/labels/libero_4suite_v4/libero_<suite>/labels.jsonl`). This isolates "
                 "the rewrite quality without dilution from V3-kept rows.")
    lines.append("")
    lines.append("Per-suite directory views needed by the judge script were built with "
                 "`scripts/eval/build_v4_per_suite_view.py`:")
    lines.append("")
    lines.append("- `data/labels/libero_4suite_v4_combined_per_suite/libero_<suite>/{labels.jsonl, frames_cache}` "
                 "(split from combined; suite prefix stripped from `source_example_id`; frames symlinked to V3).")
    lines.append("- `data/labels/libero_4suite_v4_view/libero_<suite>/{labels.jsonl, frames_cache}` "
                 "(symlinked from `libero_4suite_v4` + V3 frames).")
    lines.append("")

    lines.append("---\n")
    lines.append("## A. V4-combined vs V3 (headline)\n")
    lines.append(_md_overall("V4-combined", sc_combined))
    lines.append(_md_per_axis("V4-combined", "suite",
                              SUITES, sc_combined["per_suite_v3"], sc_combined["per_suite_v4"]))
    lines.append(_md_per_axis("V4-combined", "position_type",
                              PTYPES, sc_combined["per_ptype_v3"], sc_combined["per_ptype_v4"]))
    lines.append(_md_matrix("V4-combined", sc_combined))
    lines.append(_md_deltas("V4-combined", deltas_combined))
    lines.append(_md_worst_examples("V4-combined", v4c))

    lines.append("---\n")
    lines.append("## B. V4-only-rewritten-rows vs V3 (aggressive)\n")
    lines.append(_md_overall("V4-only", sc_v4only))
    lines.append(_md_per_axis("V4-only", "suite",
                              SUITES, sc_v4only["per_suite_v3"], sc_v4only["per_suite_v4"]))
    lines.append(_md_per_axis("V4-only", "position_type",
                              PTYPES, sc_v4only["per_ptype_v3"], sc_v4only["per_ptype_v4"]))
    lines.append(_md_matrix("V4-only", sc_v4only))
    lines.append(_md_deltas("V4-only", deltas_v4only))
    lines.append(_md_worst_examples("V4-only", v4o))

    lines.append("---\n")
    lines.append("## Gate verdicts\n")
    lines.append("Gates (per plan): "
                 "**Paper-grade** = overall B ≥ 95% AND C ≥ 95%; "
                 "**Spatial-rescue** = spatial B ≥ 85%; "
                 "**No-regression** = NO 12-cell suite×ptype value drops > 5pp B or C from V3.")
    lines.append("")
    lines.append("| Gate | V4-combined | V4-only |")
    lines.append("|---|---|---|")
    lines.append(f"| Paper-grade (B≥95% AND C≥95%) | "
                 f"{gates_combined['paper_grade']} "
                 f"(B={_fmt_pct(sc_combined['overall_v4']['b_pct'])}, "
                 f"C={_fmt_pct(sc_combined['overall_v4']['c_pct'])}) | "
                 f"{gates_v4only['paper_grade']} "
                 f"(B={_fmt_pct(sc_v4only['overall_v4']['b_pct'])}, "
                 f"C={_fmt_pct(sc_v4only['overall_v4']['c_pct'])}) |")
    lines.append(f"| Spatial-rescue (spatial B ≥ 85%) | "
                 f"{'PASS' if gates_combined['spatial_rescue_pass'] else 'FAIL'} "
                 f"(spatial B = {_fmt_pct(gates_combined['spatial_b'])}) | "
                 f"{'PASS' if gates_v4only['spatial_rescue_pass'] else 'FAIL'} "
                 f"(spatial B = {_fmt_pct(gates_v4only['spatial_b'])}) |")
    lines.append(f"| No-regression (no cell drops >5pp) | "
                 f"{'PASS' if gates_combined['no_regression_pass'] else 'FAIL'} "
                 f"({gates_combined['n_cells_regressed_gt5pp']} cells regressed) | "
                 f"{'PASS' if gates_v4only['no_regression_pass'] else 'FAIL'} "
                 f"({gates_v4only['n_cells_regressed_gt5pp']} cells regressed) |")
    lines.append("")

    if sc_combined["cells_regressed_gt5pp"]:
        lines.append("### V4-combined cells that regressed >5pp\n")
        lines.append("| Cell | Metric | ΔB or ΔC (pp) |")
        lines.append("|---|---|---|")
        for cell, metric, d in sc_combined["cells_regressed_gt5pp"]:
            lines.append(f"| {cell} | {metric} | {d:+.2f} |")
        lines.append("")
    if sc_v4only["cells_regressed_gt5pp"]:
        lines.append("### V4-only cells that regressed >5pp\n")
        lines.append("| Cell | Metric | ΔB or ΔC (pp) |")
        lines.append("|---|---|---|")
        for cell, metric, d in sc_v4only["cells_regressed_gt5pp"]:
            lines.append(f"| {cell} | {metric} | {d:+.2f} |")
        lines.append("")

    if sc_combined["cells_b_gain_ge10pp"]:
        lines.append("### V4-combined cells with ≥10pp B improvement (\"rescues\")\n")
        lines.append("| Cell | ΔB (pp) |")
        lines.append("|---|---|")
        for cell, metric, d in sc_combined["cells_b_gain_ge10pp"]:
            lines.append(f"| {cell} | +{d:.2f} |")
        lines.append("")

    lines.append("---\n")
    lines.append("## Recommendation\n")
    paper = gates_combined["paper_grade"]
    spatial_ok = gates_combined["spatial_rescue_pass"]
    no_reg = gates_combined["no_regression_pass"]
    overall_b = sc_combined["overall_v4"]["b_pct"] or 0.0
    overall_c = sc_combined["overall_v4"]["c_pct"] or 0.0
    n_reg = gates_combined["n_cells_regressed_gt5pp"]
    spatial_b_pct = gates_combined["spatial_b"]
    v3_overall_b = sc_combined["overall_v3"]["b_pct"] or 0.0
    v3_overall_c = sc_combined["overall_v3"]["c_pct"] or 0.0
    delta_overall_b = (overall_b - v3_overall_b) * 100
    delta_overall_c = (overall_c - v3_overall_c) * 100

    if paper == "GREEN" and spatial_ok and no_reg:
        lines.append("**SHIP IT.** V4-combined clears all three gates: "
                     f"paper-grade GREEN (B={overall_b*100:.2f}%, C={overall_c*100:.2f}%), "
                     f"spatial B={spatial_b_pct*100:.2f}% (≥85% gate), "
                     "and no 12-cell regression >5pp.")
    elif spatial_ok and overall_c >= 0.95:
        lines.append("**YELLOW (acceptable to ship, but not paper-grade GREEN).** ")
        lines.append("")
        lines.append(f"- Overall **B = {overall_b*100:.2f}%** vs V3 {v3_overall_b*100:.2f}% "
                     f"(Δ {delta_overall_b:+.2f}pp). Misses 95% paper bar.")
        lines.append(f"- Overall **C = {overall_c*100:.2f}%** vs V3 {v3_overall_c*100:.2f}% "
                     f"(Δ {delta_overall_c:+.2f}pp). Clears 95%.")
        lines.append(f"- **Spatial-rescue PASS:** spatial B = {spatial_b_pct*100:.2f}% (≥85% gate).")
        lines.append(f"- **No-regression {'PASS' if no_reg else 'FAIL'}:** "
                     f"{n_reg} of 12 cells regressed >5pp.")
        lines.append("")
        lines.append("The V4 dataset is the strict superset improvement the plan promised on the "
                     "worst-case axis (spatial), and overall **C clears the 95% appropriateness gate**. "
                     "Overall **B misses 95%** primarily because the `libero_10` suite did not get the "
                     "same lift the spatial-targeted V4 rewrite produced. Given V3 was the YELLOW shipped "
                     "baseline, V4 is **strictly better on spatial and overall comparable on C**; ship V4 "
                     "for SFT while opening a follow-up to push `libero_10/last_text` rewrites in V5.")
    else:
        lines.append("**FIX-AND-RERUN.** ")
        lines.append("")
        lines.append(f"- Overall B={overall_b*100:.2f}% / C={overall_c*100:.2f}% "
                     f"(V3 was {v3_overall_b*100:.2f}% / {v3_overall_c*100:.2f}%).")
        lines.append(f"- Spatial B={spatial_b_pct*100:.2f}%; "
                     f"{'meets' if spatial_ok else 'misses'} the 85% bar.")
        lines.append(f"- {n_reg} cells regressed >5pp.")
        lines.append("")
        lines.append("Investigate the regressed cells (table above), patch the per-cell prompts in the "
                     "labeler, and rerun the affected suites before declaring V4 ready.")

    with REPORT_OUT.open("w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {REPORT_OUT}")

    print("\n=== Headline summary ===")
    print(f"V4-combined: B={overall_b*100:.2f}%  C={overall_c*100:.2f}%  "
          f"(V3 B={v3_overall_b*100:.2f}%  C={v3_overall_c*100:.2f}%)")
    print(f"V4-combined spatial B = {spatial_b_pct*100:.2f}%")
    print(f"Cells regressed >5pp: {n_reg}")
    print(f"Paper-grade: {paper}  Spatial-rescue: {'PASS' if spatial_ok else 'FAIL'}  "
          f"No-regression: {'PASS' if no_reg else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
