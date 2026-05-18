#!/usr/bin/env python
"""Side-by-side scorecard diff (baseline vs candidate -> Markdown).

Consumes two scorecard JSON files (e.g. v3_scorecard.json and
v4_extraction_scorecard.json), recursively walks both to collect every leaf
numeric value, computes ``candidate - baseline`` deltas, groups results by
top-level scorecard key (e.g. ``metrics``, ``rankings``, ``config``,
``sources``), and emits a Markdown table per group plus a one-line summary.

Design notes
------------
* **Pure stdlib, CPU-only.** No numpy, no LLM, no torch. The whole script is
  ``argparse + json + re``.
* **Path normalisation.** Lists of dicts that carry a stable identifier
  (``name`` for V3 metric rows, ``config_key`` for V4 ranking rows) are
  flattened by *name* rather than by *index* so that the same metric in two
  scorecards collides on the same path regardless of list position.
* **Booleans are skipped.** ``isinstance(True, int)`` is True in Python; we
  filter ``bool`` explicitly so flags like ``higher_is_better`` and
  ``required_for_overall`` don't pollute the diff.
* **Direction heuristic.** Per the A8 spec: paths containing any of
  ``fve / cosine / accuracy / pass / margin / f1 / success`` are higher-is-
  better; paths containing ``mse / loss / error / collapse`` are lower-is-
  better; everything else is reported as a raw delta (verdict = NEUTRAL).
* **Missing keys are first-class.** A path present in baseline but absent
  from candidate is rendered as a row with an em-dash and verdict
  ``MISSING in candidate`` (never a crash).

Usage::

    python scripts/eval/diff_scorecards.py \\
        --baseline  data/sft/libero_4suite_v3/v3_scorecard.json \\
        --candidate data/sft/libero_4suite_v3/v3_scorecard.json \\
        [--output   diff.md]

Self-diff against the same file is the smoke test: every row must be NEUTRAL
with delta = 0.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# Direction heuristic
# ---------------------------------------------------------------------------

HIGHER_BETTER_KEYS: tuple[str, ...] = (
    "fve", "cosine", "accuracy", "pass", "margin", "f1", "success",
)
LOWER_BETTER_KEYS: tuple[str, ...] = (
    "mse", "loss", "error", "collapse",
)


def direction_for(path: str) -> str:
    """Return ``'higher'`` / ``'lower'`` / ``'unknown'`` for a metric path.

    Lower-is-better keywords win ties (e.g. ``mse_pass`` -> lower).
    """
    p = path.lower()
    if any(k in p for k in LOWER_BETTER_KEYS):
        return "lower"
    if any(k in p for k in HIGHER_BETTER_KEYS):
        return "higher"
    return "unknown"


# ---------------------------------------------------------------------------
# Recursive numeric walk
# ---------------------------------------------------------------------------

_INDEX_TAIL = re.compile(r"\[\d+\]$")
_TOP_KEY = re.compile(r"^([^.\[]+)")
_LABEL_KEYS = ("name", "config_key")


def _label_for(obj: dict) -> str | None:
    """If a dict carries a stable identifier, return it (else None)."""
    for k in _LABEL_KEYS:
        v = obj.get(k)
        if isinstance(v, str) and v:
            return v
    return None


def walk_numeric(obj: Any, path: str = "") -> Iterator[tuple[str, float]]:
    """Yield ``(dotted_path, value)`` for every leaf number in *obj*.

    Booleans are skipped. Lists of identified dicts (``name`` /
    ``config_key``) are flattened by the identifier rather than by index.
    """
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield path, float(obj)
        return
    if isinstance(obj, dict):
        label = _label_for(obj)
        if label is not None:
            base = _INDEX_TAIL.sub("", path)
            new_base = f"{base}.{label}" if base else label
            for k, v in obj.items():
                if k in _LABEL_KEYS:
                    continue
                yield from walk_numeric(v, f"{new_base}.{k}")
            return
        for k, v in obj.items():
            new_path = f"{path}.{k}" if path else str(k)
            yield from walk_numeric(v, new_path)
        return
    if isinstance(obj, list):
        for i, item in enumerate(obj):
            yield from walk_numeric(item, f"{path}[{i}]")
        return
    # Strings, None, etc. -> not numeric, drop silently.


def top_key(path: str) -> str:
    """First dotted segment (or first ``[``-prefixed segment) of *path*."""
    m = _TOP_KEY.match(path)
    return m.group(1) if m else path


# ---------------------------------------------------------------------------
# Verdict + rendering
# ---------------------------------------------------------------------------

def classify(delta: float, direction: str) -> str:
    """IMPROVED / REGRESSED / NEUTRAL based on delta sign and direction."""
    if delta == 0:
        return "NEUTRAL"
    if direction == "higher":
        return "IMPROVED" if delta > 0 else "REGRESSED"
    if direction == "lower":
        return "IMPROVED" if delta < 0 else "REGRESSED"
    return "NEUTRAL"


def _fmt(v: float | None) -> str:
    if v is None:
        return "—"
    if v != v:  # NaN
        return "NaN"
    if abs(v) >= 1000 or (v != 0 and abs(v) < 5e-4):
        return f"{v:.4e}"
    return f"{v:.4f}"


def _fmt_delta(d: float | None) -> str:
    if d is None:
        return "—"
    if d != d:
        return "NaN"
    if abs(d) >= 1000 or (d != 0 and abs(d) < 5e-4):
        return f"{d:+.4e}"
    return f"{d:+.4f}"


def render_diff(
    baseline: dict,
    candidate: dict,
    baseline_path: Path,
    candidate_path: Path,
) -> str:
    b = dict(walk_numeric(baseline))
    c = dict(walk_numeric(candidate))
    all_keys = sorted(set(b) | set(c))

    by_group: dict[str, list[dict[str, Any]]] = {}
    n_improved = n_regressed = n_neutral = n_missing = 0
    biggest_gain: tuple[float, str] | None = None
    biggest_loss: tuple[float, str] | None = None

    for path in all_keys:
        bv = b.get(path)
        cv = c.get(path)
        direction = direction_for(path)
        if bv is None or cv is None:
            verdict = "MISSING in candidate" if cv is None else "MISSING in baseline"
            delta: float | None = None
            n_missing += 1
        else:
            delta = cv - bv
            verdict = classify(delta, direction)
            if verdict == "IMPROVED":
                n_improved += 1
                if biggest_gain is None or abs(delta) > abs(biggest_gain[0]):
                    biggest_gain = (delta, path)
            elif verdict == "REGRESSED":
                n_regressed += 1
                if biggest_loss is None or abs(delta) > abs(biggest_loss[0]):
                    biggest_loss = (delta, path)
            else:
                n_neutral += 1

        by_group.setdefault(top_key(path), []).append({
            "path": path,
            "baseline": bv,
            "candidate": cv,
            "delta": delta,
            "direction": direction,
            "verdict": verdict,
        })

    out: list[str] = []
    out.append("# Scorecard diff")
    out.append("")
    out.append(f"- baseline:  `{baseline_path}`")
    out.append(f"- candidate: `{candidate_path}`")
    out.append("")

    for grp in sorted(by_group):
        rows = by_group[grp]
        out.append(f"## `{grp}` ({len(rows)} metric{'s' if len(rows) != 1 else ''})")
        out.append("")
        out.append("| Metric | Baseline | Candidate | Δ | Direction |")
        out.append("|---|---:|---:|---:|---|")
        for r in rows:
            out.append(
                f"| `{r['path']}` | {_fmt(r['baseline'])} | "
                f"{_fmt(r['candidate'])} | {_fmt_delta(r['delta'])} | {r['verdict']} |"
            )
        out.append("")

    out.append("## Summary")
    out.append("")
    summary = (
        f"{n_improved} improved, {n_regressed} regressed, "
        f"{n_neutral} neutral"
    )
    if n_missing:
        summary += f", {n_missing} missing"
    if biggest_gain is not None:
        summary += (
            f"; biggest gain: `{biggest_gain[1]}` ({_fmt_delta(biggest_gain[0])})"
        )
    if biggest_loss is not None:
        summary += (
            f"; biggest loss: `{biggest_loss[1]}` ({_fmt_delta(biggest_loss[0])})"
        )
    out.append(summary)
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--baseline", type=Path, required=True,
                   help="Baseline scorecard JSON.")
    p.add_argument("--candidate", type=Path, required=True,
                   help="Candidate scorecard JSON.")
    p.add_argument("--output", type=Path, default=None,
                   help="Write Markdown here (default: stdout).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    md = render_diff(baseline, candidate, args.baseline, args.candidate)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md)
    else:
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
