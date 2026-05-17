#!/usr/bin/env python
"""Side-by-side comparison of multiple LLM-judge runs.

Reads N judge JSONLs produced by ``scripts/eval/llm_judge_av_captions.py``,
aggregates by ``(variant_id, position_type)``, and emits:

    1. A single markdown table with one row per (run, variant, position) cell.
    2. A "deltas" section with pairwise differences between the FIRST run and
       each subsequent run, for matching ``(variant, position)`` keys.

Usage::

    PYTHONPATH=src python scripts/eval/compare_judge_runs.py \
        --runs v3=data/sft/libero_goal_pilot_v3/llm_judge.jsonl \
               v3_grpo=data/grpo/libero_goal_pilot_b002_judge/llm_judge.jsonl \
        --out-md data/eval/judge_comparison.md

The CLI accepts each ``--runs LABEL=PATH`` either space- or whitespace-
separated. The label may not contain ``=`` itself.

If a run is missing a particular (variant, position) cell that appears in
another run, the missing cell is reported as ``N/A`` (n=0) rather than
crashing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--runs",
        nargs="+",
        required=True,
        metavar="LABEL=PATH",
        help="One or more judge JSONLs labeled as LABEL=PATH.",
    )
    p.add_argument(
        "--out-md",
        default=None,
        help="Optional output path for the markdown report (default: stdout only).",
    )
    return p


def _parse_runs(specs: Iterable[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--runs entry {spec!r} is not LABEL=PATH")
        label, _, path = spec.partition("=")
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise SystemExit(f"--runs entry {spec!r} has empty label or path")
        out.append((label, Path(path)))
    return out


def load_judge_jsonl(path: Path) -> list[dict]:
    """Load a judge JSONL into a list of dicts, tolerating blank/garbage lines."""
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rows.append(obj)
    return rows


def _position_type_from_example_id(example_id: str) -> str | None:
    """Parse ``{src}@p{idx}_{position_type}`` -> position_type."""
    if "@p" not in example_id:
        return None
    tail = example_id.split("@p", 1)[1]
    if "_" not in tail:
        return None
    _, ptype = tail.split("_", 1)
    return ptype or None


def aggregate(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Aggregate judge rows by ``(variant_id, position_type)``.

    Returns a dict of ``{(variant, ptype): {n, n_b, n_c, b_pct, c_pct}}``.
    Rows with explicit ``error`` or missing both axes are skipped from
    pass-rate denominators (they're effectively un-graded).
    """
    by_key: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        variant = r.get("variant_id")
        if not variant:
            continue
        ptype = r.get("position_type") or _position_type_from_example_id(r.get("example_id", ""))
        ptype = ptype or "_unk"
        if (r.get("grounding") is None) and (r.get("appropriateness") is None):
            continue
        if r.get("error") and r.get("grounding") is None and r.get("appropriateness") is None:
            continue
        by_key.setdefault((variant, ptype), []).append(r)

    out: dict[tuple[str, str], dict] = {}
    for key, bucket in by_key.items():
        n = len(bucket)
        n_b = sum(1 for r in bucket if (r.get("grounding") or {}).get("verdict") == "specific")
        n_c = sum(1 for r in bucket if (r.get("appropriateness") or {}).get("verdict") == "appropriate")
        out[key] = {
            "n": n,
            "n_b_specific": n_b,
            "n_c_appropriate": n_c,
            "b_specific_pct": (n_b / n * 100.0) if n else None,
            "c_appropriate_pct": (n_c / n * 100.0) if n else None,
        }
    return out


def _fmt_pct(p: float | None) -> str:
    if p is None:
        return "N/A"
    return f"{p:.1f}"


def render_markdown(
    runs: list[tuple[str, Path]],
    aggs: list[dict[tuple[str, str], dict]],
) -> str:
    """Render the side-by-side and deltas markdown report."""
    all_keys: set[tuple[str, str]] = set()
    for agg in aggs:
        all_keys.update(agg.keys())
    sorted_keys = sorted(all_keys)

    lines: list[str] = []
    lines.append("# Judge run comparison")
    lines.append("")
    lines.append("Sources:")
    for label, path in runs:
        lines.append(f"- **{label}** -> `{path}`")
    lines.append("")
    lines.append("## Aggregate verdicts")
    lines.append("")
    lines.append("| Run | Variant | Position | n | B specific% | C appropriate% |")
    lines.append("|-----|---------|----------|---:|-----------:|---------------:|")
    for label, _ in runs:
        agg_idx = next(i for i, (lbl, _) in enumerate(runs) if lbl == label)
        agg = aggs[agg_idx]
        for variant, ptype in sorted_keys:
            cell = agg.get((variant, ptype))
            if cell is None:
                lines.append(
                    f"| {label} | {variant} | {ptype} | 0 | N/A | N/A |"
                )
            else:
                lines.append(
                    f"| {label} | {variant} | {ptype} | {cell['n']} | "
                    f"{_fmt_pct(cell['b_specific_pct'])} | "
                    f"{_fmt_pct(cell['c_appropriate_pct'])} |"
                )

    if len(runs) >= 2:
        lines.append("")
        lines.append("## Deltas vs first run")
        lines.append("")
        base_label = runs[0][0]
        base_agg = aggs[0]
        lines.append(
            f"| Run | Variant | Position | dB specific% | dC appropriate% |"
        )
        lines.append(
            f"|-----|---------|----------|------------:|----------------:|"
        )
        for label, _ in runs[1:]:
            agg_idx = next(i for i, (lbl, _) in enumerate(runs) if lbl == label)
            agg = aggs[agg_idx]
            for variant, ptype in sorted_keys:
                base = base_agg.get((variant, ptype))
                cur = agg.get((variant, ptype))
                if base is None or cur is None:
                    db = "N/A"
                    dc = "N/A"
                else:
                    if base["b_specific_pct"] is None or cur["b_specific_pct"] is None:
                        db = "N/A"
                    else:
                        db = f"{cur['b_specific_pct'] - base['b_specific_pct']:+.1f}"
                    if base["c_appropriate_pct"] is None or cur["c_appropriate_pct"] is None:
                        dc = "N/A"
                    else:
                        dc = f"{cur['c_appropriate_pct'] - base['c_appropriate_pct']:+.1f}"
                lines.append(
                    f"| {label} vs {base_label} | {variant} | {ptype} | {db} | {dc} |"
                )

    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    runs = _parse_runs(args.runs)

    aggs: list[dict[tuple[str, str], dict]] = []
    for label, path in runs:
        rows = load_judge_jsonl(path)
        aggs.append(aggregate(rows))
        if not rows:
            print(f"warn: {label} ({path}) is empty or missing", file=sys.stderr)

    md = render_markdown(runs, aggs)
    print(md)
    if args.out_md:
        out_path = Path(args.out_md)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md)
        print(f"\nwrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
