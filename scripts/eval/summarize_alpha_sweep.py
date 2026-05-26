#!/usr/bin/env python
"""Summarize a (possibly partial) alpha sweep into a markdown table.

Reads ``runs/alpha_sweep/<date>/alpha_<scale>.json`` files written by
``compare_cf_steer_checkpoints.py`` and produces:

  - A console table (matched succ, mismatched succ, Δ_cw, steer_lift, n) per α.
  - A markdown file ``summary.md`` in the same dir.
  - The Stage-0 verdict (DOSE-MISCALIBRATION / CODEC FAILURE / INCONCLUSIVE).

Safe to re-run while the sweep is still going — it ignores missing alphas
and writes whatever's available. Use this when you want a snapshot before
the wrapper's own ``summary.json`` is final.

Usage::

    PYTHONPATH=src python scripts/eval/summarize_alpha_sweep.py \\
        --sweep-dir runs/alpha_sweep/<date>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


_ALPHA_RE = re.compile(r"^alpha_(?P<scale>-?\d+(?:\.\d+)?)\.json$")


def _load_alpha_jsons(sweep_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(sweep_dir.iterdir()):
        m = _ALPHA_RE.match(p.name)
        if not m:
            continue
        try:
            obj = json.loads(p.read_text())
        except json.JSONDecodeError:
            obj = {"_error": "json decode failed"}
        obj["alpha_scale"] = float(m.group("scale"))
        obj["_path"] = str(p)
        rows.append(obj)
    rows.sort(key=lambda r: r["alpha_scale"])
    return rows


def _row_for_cond(summary: dict, cond: str) -> dict:
    """Extract the four arm rates from a compare summary.

    The compare script's _make_record_key uses the legacy short form when
    arms are at their default; the full ``{cond}__{intent}__{causal}`` form
    only appears for non-default combinations. So matched+semantic lives
    under ``{cond}_predicate_rate``, etc.
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

    def _first_present(keys):
        for k in keys:
            if k in summary:
                return summary[k]
        return None

    return {
        "matched_semantic": _first_present(matched_semantic_keys),
        "mismatched_semantic": _first_present(mismatched_semantic_keys),
        "matched_no_steer": _first_present(no_steer_keys),
    }


def _verdict(rows: list[dict]) -> str:
    deltas = [r["delta_cw"] for r in rows if isinstance(r.get("delta_cw"), (int, float))]
    if not deltas:
        return "INCONCLUSIVE (no Δ_cw rows)"
    if max(deltas) >= 0.05:
        return "DOSE-MISCALIBRATION (some α lifts matched over mismatched by ≥5pp)"
    if all(-0.02 <= d <= 0.02 for d in deltas):
        return "CODEC FAILURE (Δ_cw stays in [-2pp, +2pp] across every α)"
    return "INCONCLUSIVE (movement but no clear winner)"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sweep-dir", required=True)
    p.add_argument("--conditions", default="sft_av",
                   help="Comma list of condition keys to extract (e.g. 'sft_av,grpo_av').")
    args = p.parse_args(argv)

    sweep_dir = Path(args.sweep_dir)
    alpha_summaries = _load_alpha_jsons(sweep_dir)
    cond_list = [c.strip() for c in args.conditions.split(",") if c.strip()]

    rows: list[dict] = []
    for summ in alpha_summaries:
        for cond in cond_list:
            arms = _row_for_cond(summ, cond)
            m = arms["matched_semantic"]
            w = arms["mismatched_semantic"]
            ns = arms["matched_no_steer"]
            delta_cw = (m - w) if isinstance(m, (int, float)) and isinstance(w, (int, float)) else None
            steer_lift = (m - ns) if isinstance(m, (int, float)) and isinstance(ns, (int, float)) else None
            rows.append({
                "alpha_scale": summ["alpha_scale"],
                "condition": cond,
                "matched_semantic": m,
                "mismatched_semantic": w,
                "matched_no_steer": ns,
                "delta_cw": delta_cw,
                "steer_lift": steer_lift,
                "n": summ.get("n"),
            })

    # Console
    print(f"alpha-sweep snapshot from {sweep_dir} (n_alphas={len(alpha_summaries)})")
    print()
    header = f"{'alpha':>6s}  {'cond':<8s}  {'matched':>8s}  {'mismatched':>10s}  {'no_steer':>8s}  {'Δ_cw':>8s}  {'lift':>7s}  {'n':>3s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        def _pct(v):
            return f"{v:7.2%}" if isinstance(v, (int, float)) else "    n/a"
        def _signed(v):
            return f"{v:+7.2%}" if isinstance(v, (int, float)) else "    n/a"
        print(
            f"{r['alpha_scale']:6.3f}  {r['condition']:<8s}  {_pct(r['matched_semantic']):>8s}  "
            f"{_pct(r['mismatched_semantic']):>10s}  {_pct(r['matched_no_steer']):>8s}  "
            f"{_signed(r['delta_cw']):>8s}  {_signed(r['steer_lift']):>7s}  {str(r.get('n') or 'n/a'):>3s}"
        )

    verdict = _verdict(rows)
    print()
    print(f"Stage-0 verdict (partial-data-tolerant): {verdict}")

    # Markdown out
    md = ["# Alpha sweep snapshot", "", f"Source: `{sweep_dir}` — n_alphas={len(alpha_summaries)}", ""]
    md.append("| α | cond | matched | mismatched | no_steer | Δ_cw | steer_lift | n |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        def _md_pct(v):
            return f"{v:.2%}" if isinstance(v, (int, float)) else "n/a"
        def _md_signed(v):
            return f"{v:+.2%}" if isinstance(v, (int, float)) else "n/a"
        md.append(
            f"| {r['alpha_scale']:.3f} | `{r['condition']}` | {_md_pct(r['matched_semantic'])} | "
            f"{_md_pct(r['mismatched_semantic'])} | {_md_pct(r['matched_no_steer'])} | "
            f"{_md_signed(r['delta_cw'])} | {_md_signed(r['steer_lift'])} | {r.get('n') or 'n/a'} |"
        )
    md += ["", f"**Verdict:** {verdict}"]
    out_md = sweep_dir / "summary.md"
    out_md.write_text("\n".join(md) + "\n")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
