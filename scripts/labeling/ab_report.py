#!/usr/bin/env python
"""Render an HTML report for one A/B round.

Reads ``<round_dir>/scores.json`` and the per-variant ``labels.jsonl`` /
``grades.jsonl`` files and emits ``<round_dir>/report.html`` with:

  * per-axis pass-rate bar chart (inline SVG, no external deps)
  * per-variant summary table
  * side-by-side label rendering for the first N eval rows (default 8),
    one column per variant, with auto/LLM verdicts annotated
  * (B)-LLM and (C)-LLM top failure-mode lists per variant

The renderer is dependency-free; just stdlib + a tiny bit of SVG.

Example::

    PYTHONPATH=src python scripts/labeling/ab_report.py --round-dir data/prompt_ab/round_01
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import logging
import sys
from pathlib import Path
from typing import Iterable


def _load_scores(round_dir: Path) -> dict:
    scores_path = round_dir / "scores.json"
    if not scores_path.exists():
        raise FileNotFoundError(f"missing {scores_path}")
    return json.loads(scores_path.read_text())


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _img_data_uri(path: str) -> str:
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        ext = Path(path).suffix.lstrip(".").lower() or "jpeg"
        if ext == "jpg":
            ext = "jpeg"
        return f"data:image/{ext};base64,{b64}"
    except FileNotFoundError:
        return ""


def _bar_chart_svg(scores: dict[str, dict], threshold: float = 0.95) -> str:
    """Tiny stacked-bar chart of per-axis pass rates per variant."""
    variants = list(scores)
    if not variants:
        return ""
    bar_h = 18
    pad_y = 28
    label_w = 60
    chart_w = 520
    chart_left = label_w + 10
    n = len(variants)
    h = pad_y * 2 + n * 3 * (bar_h + 4) + n * 8

    rows = []
    rows.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{chart_left + chart_w + 60}" height="{h}" '
        'style="font-family:system-ui,Arial,sans-serif;font-size:11px;background:white;">'
    )
    # Threshold line
    tx = chart_left + int(chart_w * threshold)
    rows.append(
        f'<line x1="{tx}" y1="{pad_y - 6}" x2="{tx}" y2="{h - pad_y + 6}" '
        f'stroke="#c33" stroke-dasharray="4 3" stroke-width="1"/>'
    )
    rows.append(
        f'<text x="{tx + 3}" y="{pad_y - 8}" fill="#c33">{int(threshold * 100)}%</text>'
    )
    y = pad_y
    for v in variants:
        c = scores[v]
        rows.append(
            f'<text x="0" y="{y + bar_h - 4}" font-weight="bold">{html.escape(v)}</text>'
        )
        for axis_name, axis_key, color in (
            ("a", "pass_rate_a", "#4a90e2"),
            ("b", "pass_rate_b_combined", "#7b61ff"),
            ("c", "pass_rate_c_combined", "#3cba54"),
        ):
            rate = float(c.get(axis_key, 0.0))
            w = int(chart_w * rate)
            rows.append(
                f'<rect x="{chart_left}" y="{y}" width="{w}" height="{bar_h}" fill="{color}" />'
            )
            rows.append(
                f'<rect x="{chart_left}" y="{y}" width="{chart_w}" height="{bar_h}" '
                'fill="none" stroke="#bbb" />'
            )
            rows.append(
                f'<text x="{chart_left + chart_w + 6}" y="{y + bar_h - 4}">'
                f'{axis_name}={rate:.3f}</text>'
            )
            y += bar_h + 4
        y += 8
    rows.append("</svg>")
    return "".join(rows)


def _summary_table(scores: dict[str, dict]) -> str:
    headers = (
        "variant", "n_labels", "axis_a", "axis_b_combined", "axis_c_combined",
        "b_llm", "c_llm", "pass95",
    )
    out: list[str] = ['<table class="summary"><thead><tr>']
    for h in headers:
        out.append(f"<th>{html.escape(h)}</th>")
    out.append("</tr></thead><tbody>")
    for v, c in scores.items():
        llm = c.get("llm", {})
        out.append("<tr>")
        out.append(f"<td><b>{html.escape(v)}</b></td>")
        out.append(f"<td>{c.get('n_labels', 0)}</td>")
        out.append(f"<td>{c.get('pass_rate_a', 0):.3f}</td>")
        out.append(f"<td>{c.get('pass_rate_b_combined', 0):.3f}</td>")
        out.append(f"<td>{c.get('pass_rate_c_combined', 0):.3f}</td>")
        out.append(f"<td>{llm.get('pass_rate_b_llm', 0):.3f}</td>")
        out.append(f"<td>{llm.get('pass_rate_c_llm', 0):.3f}</td>")
        bg = "#cfeac6" if c.get("passes_95") else "#fcd5d5"
        out.append(
            f'<td style="background:{bg};">{("YES" if c.get("passes_95") else "no")}</td>'
        )
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def _top_failures(scores: dict[str, dict]) -> str:
    out: list[str] = ['<div class="failures">']
    for v, c in scores.items():
        auto = c.get("auto", {})
        llm = c.get("llm", {})
        out.append(f"<details><summary><b>{html.escape(v)}</b> failure modes</summary>")
        for axis_label, failures in (
            ("(a) format", auto.get("axis_a", {}).get("top_failures", [])),
            ("(b) auto distinctness", auto.get("axis_b_auto", {}).get("top_failures", [])),
            ("(c) auto vocab", auto.get("axis_c_auto", {}).get("top_failures", [])),
            ("(b) LLM grounding", llm.get("top_b_failures", [])),
            ("(c) LLM appropriateness", llm.get("top_c_failures", [])),
        ):
            if not failures:
                continue
            out.append(f"<p style='margin:6px 0;'><i>{html.escape(axis_label)}:</i></p><ul>")
            for reason, count in failures[:6]:
                out.append(
                    f"<li><code>{html.escape(str(reason))}</code> &times; {count}</li>"
                )
            out.append("</ul>")
        out.append("</details>")
    out.append("</div>")
    return "".join(out)


def _side_by_side(
    round_dir: Path,
    variants: list[str],
    eval_set_path: Path,
    n_examples: int = 8,
    embed_images: bool = False,
) -> str:
    """Render a side-by-side grid: rows are eval examples, columns are variants."""
    # Load eval set (for instruction + image paths)
    eval_rows = _load_jsonl(eval_set_path)
    # Stratify the rendered subset for visual coverage.
    by_type: dict[str, list[dict]] = {}
    for r in eval_rows:
        by_type.setdefault(r["position_type"], []).append(r)
    n_per_type = max(1, n_examples // 3)
    pick: list[dict] = []
    for ptype in ("last_text", "image_patch", "anchor"):
        pick.extend(by_type.get(ptype, [])[:n_per_type])

    # Load each variant's labels.jsonl into a dict[eval_id] -> row.
    variant_labels: dict[str, dict[str, dict]] = {}
    variant_grades: dict[str, dict[str, dict]] = {}
    for v in variants:
        vdir = round_dir / f"variant_{v}"
        lbl = {r["example_id"]: r for r in _load_jsonl(vdir / "labels.jsonl")}
        grd = {r["example_id"]: r for r in _load_jsonl(vdir / "grades.jsonl")}
        variant_labels[v] = lbl
        variant_grades[v] = grd

    out: list[str] = ['<table class="examples">']
    out.append("<thead><tr><th>eval row</th>")
    for v in variants:
        out.append(f"<th>{html.escape(v)}</th>")
    out.append("</tr></thead><tbody>")
    for r in pick:
        out.append("<tr>")
        instr = html.escape(r.get("instruction") or "(no instruction)")
        pos = html.escape(r["position_type"])
        eid = html.escape(r["eval_id"])
        out.append(
            f'<td class="evalcell"><div class="meta"><b>{pos}</b><br>'
            f'{eid}<br><i>{instr}</i></div>'
        )
        for ipath in r["image_paths"][:1]:  # show only first image for compactness
            if embed_images:
                uri = _img_data_uri(ipath)
                if uri:
                    out.append(
                        f'<img class="frame" src="{uri}" alt="frame" />'
                    )
            else:
                rel = Path(ipath).resolve()
                out.append(
                    f'<img class="frame" src="file://{rel}" alt="frame" />'
                )
        out.append("</td>")
        for v in variants:
            lbl = variant_labels[v].get(r["eval_id"], {})
            grd = variant_grades[v].get(r["eval_id"], {})
            desc = lbl.get("description", "")
            cell_class = "label"
            if lbl.get("error"):
                desc = f"[error: {lbl.get('error')}]"
                cell_class = "label error"
            g = grd.get("grounding") or {}
            a = grd.get("appropriateness") or {}
            chip_grounding = (
                f'<span class="chip {"pass" if g.get("verdict") == "specific" else "fail"}">'
                f'b={html.escape(str(g.get("verdict") or "?"))}</span>'
            )
            chip_appro = (
                f'<span class="chip {"pass" if a.get("verdict") == "appropriate" else "fail"}">'
                f'c={html.escape(str(a.get("verdict") or "?"))}</span>'
            )
            out.append(
                f'<td class="{cell_class}">{chip_grounding} {chip_appro}'
                f'<pre>{html.escape(desc)}</pre></td>'
            )
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>
  body {{ font-family: system-ui, Arial, sans-serif; max-width: 1400px;
          margin: 24px auto; padding: 0 16px; color: #222; }}
  h1, h2 {{ margin: 0.6em 0 0.4em; }}
  .summary {{ border-collapse: collapse; margin: 1em 0; }}
  .summary th, .summary td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: right; }}
  .summary th {{ background: #f3f3f3; }}
  .summary td:first-child, .summary th:first-child {{ text-align: left; }}
  .examples {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
  .examples td, .examples th {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
  .examples th {{ background: #f3f3f3; }}
  .examples .evalcell {{ width: 250px; }}
  .examples .frame {{ max-width: 240px; max-height: 180px; }}
  .examples .label pre {{ white-space: pre-wrap; word-wrap: break-word;
                          font-family: ui-monospace, Menlo, Consolas, monospace;
                          font-size: 12px; margin: 6px 0 0; }}
  .examples .label.error {{ background: #ffe7e7; }}
  .chip {{ display: inline-block; padding: 2px 6px; border-radius: 8px;
           font-size: 11px; color: white; }}
  .chip.pass {{ background: #3cba54; }}
  .chip.fail {{ background: #c33; }}
  .failures details {{ margin: 6px 0; }}
  .failures li {{ font-size: 12px; margin: 1px 0; }}
  pre.scores {{ background: #f7f7f7; padding: 12px; border-radius: 6px;
                font-size: 12px; }}
</style></head><body>
<h1>{title}</h1>
<p>Threshold: <b>{threshold:.2f}</b>. Round dir: <code>{round_dir}</code></p>
<h2>Per-axis pass rates</h2>
{chart}
<h2>Summary</h2>
{summary}
<h2>Failure modes</h2>
{failures}
<h2>Side-by-side examples ({n_examples})</h2>
{side_by_side}
<h2>Raw scorecards</h2>
<pre class="scores">{raw_scores}</pre>
</body></html>
"""


def render(
    round_dir: Path,
    eval_set: Path,
    output_html: Path | None = None,
    n_examples: int = 9,
    embed_images: bool = False,
) -> Path:
    scores = _load_scores(round_dir)
    threshold = max(
        (s.get("pass_threshold", 0.95) for s in scores.values()), default=0.95,
    )
    variants = list(scores.keys())

    title = f"Prompt A/B round {round_dir.name.replace('round_', '')}"
    chart = _bar_chart_svg(scores, threshold=threshold)
    summary = _summary_table(scores)
    failures = _top_failures(scores)
    side_by_side = _side_by_side(
        round_dir, variants, eval_set,
        n_examples=n_examples, embed_images=embed_images,
    )

    html_str = HTML_TEMPLATE.format(
        title=html.escape(title),
        threshold=threshold,
        round_dir=html.escape(str(round_dir)),
        chart=chart,
        summary=summary,
        failures=failures,
        n_examples=n_examples,
        side_by_side=side_by_side,
        raw_scores=html.escape(json.dumps(scores, indent=2, default=str)),
    )
    output_html = output_html or (round_dir / "report.html")
    output_html.write_text(html_str)
    return output_html


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--round-dir", required=True)
    p.add_argument("--eval-set", default="data/prompt_ab/eval_set.jsonl")
    p.add_argument("--output", default=None,
                   help="Output HTML path (default <round-dir>/report.html)")
    p.add_argument("--n-examples", type=int, default=9,
                   help="Number of side-by-side examples to render (default 9, 3/type).")
    p.add_argument("--embed-images", action="store_true",
                   help="Embed images as data URIs (larger file, but portable).")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level)
    out = render(
        round_dir=Path(args.round_dir),
        eval_set=Path(args.eval_set),
        output_html=Path(args.output) if args.output else None,
        n_examples=args.n_examples,
        embed_images=args.embed_images,
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
