#!/usr/bin/env python
"""Helpers for ``scripts/eval/run_post_sft_evals_v4.sh``.

Two pure-python utilities the wrapper invokes (so the bash side stays
readable and free of inline python heredocs):

* ``write_summary`` — read the V4 SFT scorecard JSON (same schema as
  ``build_v3_scorecard.py`` produces) plus optional extraction-scorecard
  JSON, and write a one-line ``PASS|WARN|FAIL`` ``SUMMARY.txt`` with a
  small block of key metrics underneath. Designed to be greppable by CI.

* ``check_gpu`` — best-effort "does this box have a usable CUDA GPU"
  probe. Returns 0 if yes, non-zero otherwise. Faster + more accurate
  than ``nvidia-smi`` alone because it also catches the
  ``CUDA_VISIBLE_DEVICES=""`` masking case the wrapper supports.

Neither helper modifies state outside its ``--out`` path. They never
import any other eval script, so adding them does not change V3/V4
eval logic.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# write_summary
# ---------------------------------------------------------------------------

_KEY_METRICS = (
    "retrieval_margin",
    "retrieval_at_1",
    "judge_grounding_specific_pct",
    "judge_anti_template_specific_pct",
    "sim_correct_minus_wrong",
    "sim_correct_success",
    "closed_greedy_cosine",
)


def _fmt_metric(row: dict[str, Any]) -> str:
    v = row.get("value")
    verdict = row.get("verdict", "NA")
    if not isinstance(v, (int, float)):
        v_str = "    N/A"
    else:
        v_str = f"{v:8.4f}" if abs(v) < 100 else f"{v:8.2f}"
    return f"{row.get('name', '?'):<40s} {v_str}  [{verdict}]"


def _cmd_write_summary(args: argparse.Namespace) -> int:
    scorecard_p = Path(args.scorecard)
    if not scorecard_p.exists():
        # No scorecard means something upstream did not run; emit a FAIL
        # summary so CI / parent shell can grep for it.
        summary = "OVERALL=FAIL reason=missing_scorecard"
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(summary + "\n")
        print(summary)
        return 0

    sc = json.loads(scorecard_p.read_text())
    overall = sc.get("overall", "FAIL")
    rows = sc.get("metrics") or []
    by_name = {r.get("name"): r for r in rows if isinstance(r, dict)}

    # Headline: greppable single line.
    parts = [f"OVERALL={overall}"]
    for name in _KEY_METRICS:
        row = by_name.get(name)
        if row is None:
            parts.append(f"{name}=NA")
            continue
        v = row.get("value")
        verdict = row.get("verdict", "NA")
        if isinstance(v, (int, float)):
            parts.append(f"{name}={v:.4f}/{verdict}")
        else:
            parts.append(f"{name}=NA/{verdict}")

    # Optional: surface the extraction-scorecard winner if it was produced.
    ext_winner_line: str | None = None
    if args.extraction_scorecard:
        ext_p = Path(args.extraction_scorecard)
        if ext_p.exists():
            try:
                ext = json.loads(ext_p.read_text())
                dec = ext.get("decision") or {}
                winner = dec.get("winner") or {}
                rec = dec.get("recommendation", "?")
                if winner:
                    parts.append(
                        f"extraction_winner=L{winner.get('layer','?')}__"
                        f"{winner.get('strategy','?')}/{rec}"
                    )
                    ext_winner_line = (
                        f"  extraction winner: layer={winner.get('layer','?')} "
                        f"strategy={winner.get('strategy','?')} "
                        f"rank_score={winner.get('rank_score','?'):+.3f} "
                        f"recommendation={rec}"
                    )
            except Exception:
                pass

    headline = " ".join(parts)

    # Multi-line body (after the headline) so a human reader gets the
    # full per-metric table without having to open the JSON.
    body_lines = [headline, "", "Per-metric verdicts:"]
    for name in _KEY_METRICS:
        row = by_name.get(name)
        if row is None:
            body_lines.append(f"  {name:<40s}     N/A  [NA]")
            continue
        body_lines.append("  " + _fmt_metric(row))
    if ext_winner_line is not None:
        body_lines.append("")
        body_lines.append(ext_winner_line)
    body_lines.append("")
    body_lines.append(f"scorecard: {scorecard_p}")
    body = "\n".join(body_lines)

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(body + "\n")

    print(headline)
    return 0


# ---------------------------------------------------------------------------
# check_gpu
# ---------------------------------------------------------------------------

def _cmd_check_gpu(args: argparse.Namespace) -> int:
    """Return 0 iff torch.cuda reports at least one device."""
    try:
        import torch  # noqa: WPS433  (imports inside fn to keep CLI fast)
    except Exception as e:
        print(f"no_gpu: torch import failed ({e!s})")
        return 1
    try:
        avail = bool(torch.cuda.is_available()) and torch.cuda.device_count() > 0
    except Exception as e:
        print(f"no_gpu: torch.cuda probe raised ({e!s})")
        return 1
    if not avail:
        print("no_gpu: torch.cuda.is_available() is False")
        return 1
    print(f"gpu: {torch.cuda.device_count()} device(s)")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    ws = sub.add_parser("write_summary",
                        help="Render SUMMARY.txt from a V4 SFT scorecard JSON.")
    ws.add_argument("--scorecard", required=True,
                    help="Path to v4_sft_scorecard.json (or v3_scorecard.json).")
    ws.add_argument("--extraction-scorecard", default=None,
                    help="Optional path to v4_extraction_scorecard.json.")
    ws.add_argument("--out", required=True, help="Output SUMMARY.txt path.")
    ws.set_defaults(func=_cmd_write_summary)

    gp = sub.add_parser("check_gpu",
                        help="Exit 0 iff torch.cuda reports at least one device.")
    gp.set_defaults(func=_cmd_check_gpu)

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
