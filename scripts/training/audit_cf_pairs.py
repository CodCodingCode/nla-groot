#!/usr/bin/env python
"""Scorecard + quality gate for counterfactual pairs JSONL files.

Reads the JSONL produced by ``mine_grpo_counterfactual_pairs.py``, computes
balance / coverage / resolvability stats, and writes:

    <pairs>.audit.json   - machine-readable scorecard
    <pairs>.audit.md     - 1-page summary with src x tgt matrix

When called with ``--gate``, exit code is 0 iff every gate criterion is met,
otherwise 1. The CLI prints a one-line ``PASS`` / ``FAIL: <reasons>`` to
stdout so a bash driver can grep it.

Gate (all must hold):

  - Every canonical task in
    :data:`nla.eval.steerability.predicates.GOAL_TASKS` appears as a
    ``source_task`` AND as a ``target_task`` at least once.
  - No source-task share is < 5% or > 25%; same for target-task share.
  - ``counterfactual_fraction`` in [0.45, 0.55].
  - Less than 2% of rows have unresolvable ``source_task`` or ``target_task``.

Reported but NOT gated this round:

  - ``position_type`` mix.
  - Episode coverage (max/min/mean rows per episode).
  - Step-index histogram (early/mid/late buckets).
  - Pair (src, tgt) coverage matrix.

Pure-CPU, no LLM, runs in well under a second on a 5k-row file.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from nla.eval.steerability.predicates import GOAL_TASKS  # noqa: E402


logger = logging.getLogger("audit_cf_pairs")


# Gate thresholds. Kept as module-level constants so the tuner helper can
# reuse the same numbers without re-parsing argparse.
GATE_MIN_PCT = 0.05
GATE_MAX_PCT = 0.25
GATE_CF_MIN = 0.45
GATE_CF_MAX = 0.55
GATE_MAX_UNRESOLVABLE_PCT = 0.02


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------


def _read_pairs(path: Path) -> list[dict]:
    rows: list[dict] = []
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


def _pct(counter: Counter[str], total: int) -> dict[str, float]:
    if total <= 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def _step_bucket(step_index: int, ep_max: int) -> str:
    if ep_max <= 0:
        return "unknown"
    frac = step_index / float(ep_max)
    if frac < 0.25:
        return "early_0_25"
    if frac < 0.50:
        return "mid_25_50"
    if frac < 0.75:
        return "mid_50_75"
    return "late_75_100"


def build_scorecard(rows: list[dict]) -> dict[str, Any]:
    n_rows = len(rows)
    all_canon = list(GOAL_TASKS.keys())

    src_counts: Counter[str] = Counter()
    tgt_counts: Counter[str] = Counter()
    pos_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    seen_source_ids: set[str] = set()
    rows_per_episode: Counter[int] = Counter()
    rows_per_source_id: Counter[str] = Counter()
    n_counterfactual = 0
    n_unresolvable_src = 0
    n_unresolvable_tgt = 0

    ep_to_max_step: dict[int, int] = defaultdict(int)
    step_records: list[tuple[int, int]] = []  # (episode_index, step_index)

    for r in rows:
        src = r.get("source_task") or ""
        tgt = r.get("target_task") or ""
        if src not in GOAL_TASKS:
            n_unresolvable_src += 1
        else:
            src_counts[src] += 1
        if tgt not in GOAL_TASKS:
            n_unresolvable_tgt += 1
        else:
            tgt_counts[tgt] += 1
        if src in GOAL_TASKS and tgt in GOAL_TASKS:
            pair_counts[(src, tgt)] += 1
        if bool(r.get("is_counterfactual", False)):
            n_counterfactual += 1
        ptype = r.get("position_type")
        if ptype:
            pos_counts[str(ptype)] += 1
        sid = r.get("source_example_id")
        if sid:
            seen_source_ids.add(str(sid))
            rows_per_source_id[str(sid)] += 1
        ep = r.get("episode_index")
        st = r.get("step_index")
        if isinstance(ep, int):
            rows_per_episode[ep] += 1
            if isinstance(st, int):
                ep_to_max_step[ep] = max(ep_to_max_step[ep], int(st))
                step_records.append((int(ep), int(st)))

    step_hist: Counter[str] = Counter()
    for ep, st in step_records:
        step_hist[_step_bucket(st, ep_to_max_step.get(ep, 0))] += 1

    rpe = list(rows_per_episode.values())
    n_unique_pairs = len(pair_counts)
    n_dup_rows = n_rows - len(seen_source_ids)

    scorecard: dict[str, Any] = {
        "n_rows": n_rows,
        "n_unique_source_ids": len(seen_source_ids),
        "n_unique_pairs": n_unique_pairs,
        "n_pairs_possible": len(GOAL_TASKS) * len(GOAL_TASKS),
        "dup_rate": (n_dup_rows / n_rows) if n_rows > 0 else 0.0,

        "counterfactual_fraction": (n_counterfactual / n_rows) if n_rows > 0 else 0.0,
        "n_counterfactual": int(n_counterfactual),

        "source_task_counts": dict(src_counts),
        "source_task_pct": _pct(src_counts, sum(src_counts.values())),
        "target_task_counts": dict(tgt_counts),
        "target_task_pct": _pct(tgt_counts, sum(tgt_counts.values())),

        "position_type_counts": dict(pos_counts),
        "position_type_pct": _pct(pos_counts, sum(pos_counts.values())),

        "episode_coverage": {
            "n_episodes": len(rows_per_episode),
            "min_rows_per_episode": (min(rpe) if rpe else 0),
            "max_rows_per_episode": (max(rpe) if rpe else 0),
            "mean_rows_per_episode": (statistics.mean(rpe) if rpe else 0.0),
            "median_rows_per_episode": (statistics.median(rpe) if rpe else 0.0),
        },

        "step_distribution": dict(step_hist),

        "unresolvable_source_pct": (n_unresolvable_src / n_rows) if n_rows > 0 else 0.0,
        "unresolvable_target_pct": (n_unresolvable_tgt / n_rows) if n_rows > 0 else 0.0,

        "pair_matrix": {
            f"{s}__VS__{t}": int(c) for (s, t), c in pair_counts.items()
        },

        "all_canon_tasks": sorted(all_canon),
    }

    # Gate evaluation.
    failures: list[str] = []
    src_pct = scorecard["source_task_pct"]
    tgt_pct = scorecard["target_task_pct"]

    missing_src = [t for t in all_canon if t not in src_counts]
    if missing_src:
        failures.append(f"source missing tasks: {missing_src}")
    missing_tgt = [t for t in all_canon if t not in tgt_counts]
    if missing_tgt:
        failures.append(f"target missing tasks: {missing_tgt}")

    over_src = sorted([t for t, p in src_pct.items() if p > GATE_MAX_PCT])
    if over_src:
        failures.append(
            f"source over {int(GATE_MAX_PCT*100)}%: " +
            ", ".join(f"{t}={src_pct[t]:.1%}" for t in over_src)
        )
    under_src = sorted([
        t for t, p in src_pct.items()
        if p < GATE_MIN_PCT and t in src_counts
    ])
    if under_src:
        failures.append(
            f"source under {int(GATE_MIN_PCT*100)}%: " +
            ", ".join(f"{t}={src_pct[t]:.1%}" for t in under_src)
        )
    over_tgt = sorted([t for t, p in tgt_pct.items() if p > GATE_MAX_PCT])
    if over_tgt:
        failures.append(
            f"target over {int(GATE_MAX_PCT*100)}%: " +
            ", ".join(f"{t}={tgt_pct[t]:.1%}" for t in over_tgt)
        )
    under_tgt = sorted([
        t for t, p in tgt_pct.items()
        if p < GATE_MIN_PCT and t in tgt_counts
    ])
    if under_tgt:
        failures.append(
            f"target under {int(GATE_MIN_PCT*100)}%: " +
            ", ".join(f"{t}={tgt_pct[t]:.1%}" for t in under_tgt)
        )

    cf = scorecard["counterfactual_fraction"]
    if not (GATE_CF_MIN <= cf <= GATE_CF_MAX):
        failures.append(
            f"counterfactual_fraction {cf:.3f} outside "
            f"[{GATE_CF_MIN}, {GATE_CF_MAX}]"
        )

    if scorecard["unresolvable_source_pct"] > GATE_MAX_UNRESOLVABLE_PCT:
        failures.append(
            f"unresolvable_source_pct {scorecard['unresolvable_source_pct']:.3%} "
            f"> {GATE_MAX_UNRESOLVABLE_PCT:.0%}"
        )
    if scorecard["unresolvable_target_pct"] > GATE_MAX_UNRESOLVABLE_PCT:
        failures.append(
            f"unresolvable_target_pct {scorecard['unresolvable_target_pct']:.3%} "
            f"> {GATE_MAX_UNRESOLVABLE_PCT:.0%}"
        )

    scorecard["gate_pass"] = (not failures)
    scorecard["gate_failures"] = failures
    return scorecard


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _fmt_pct(p: float) -> str:
    return f"{100.0 * p:.1f}%"


def _render_task_table(
    title: str,
    counts: dict[str, int],
    pcts: dict[str, float],
    all_canon: list[str],
) -> str:
    lines = [f"### {title}", "", "| Task | Count | Share |", "|---|---:|---:|"]
    for t in all_canon:
        c = counts.get(t, 0)
        p = pcts.get(t, 0.0)
        lines.append(f"| `{t}` | {c} | {_fmt_pct(p)} |")
    return "\n".join(lines) + "\n"


def _render_pair_matrix(
    pair_counts_flat: dict[str, int],
    all_canon: list[str],
) -> str:
    # Re-expand the "src__VS__tgt" keys to a square table.
    mat: dict[tuple[str, str], int] = {}
    for k, v in pair_counts_flat.items():
        s, _, t = k.partition("__VS__")
        mat[(s, t)] = int(v)
    lines = ["### Pair coverage matrix (rows = source, cols = target)", ""]
    header = "| src \\ tgt | " + " | ".join(t[:14] for t in all_canon) + " |"
    sep = "|" + "|".join(["---"] * (len(all_canon) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for s in all_canon:
        cells = []
        for t in all_canon:
            v = mat.get((s, t), 0)
            cells.append(str(v) if v > 0 else " ")
        lines.append(f"| `{s[:24]}` | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_markdown(scorecard: dict[str, Any], pairs_path: Path) -> str:
    all_canon = scorecard["all_canon_tasks"]
    head = [
        f"# CF Pairs Audit: `{pairs_path.name}`",
        "",
        f"- Path: `{pairs_path}`",
        f"- Rows: {scorecard['n_rows']}",
        f"- Unique source_example_ids: {scorecard['n_unique_source_ids']}",
        f"- Unique (src, tgt) pairs: {scorecard['n_unique_pairs']} "
        f"/ {scorecard['n_pairs_possible']}",
        f"- Dup rate: {_fmt_pct(scorecard['dup_rate'])}",
        f"- Counterfactual fraction: {scorecard['counterfactual_fraction']:.3f} "
        f"(target [{GATE_CF_MIN}, {GATE_CF_MAX}])",
        f"- Unresolvable source / target: "
        f"{_fmt_pct(scorecard['unresolvable_source_pct'])} / "
        f"{_fmt_pct(scorecard['unresolvable_target_pct'])}",
        "",
    ]
    if scorecard["gate_pass"]:
        head.append("**GATE: PASS**\n")
    else:
        head.append("**GATE: FAIL**\n")
        head.append("Failures:")
        for f in scorecard["gate_failures"]:
            head.append(f"- {f}")
        head.append("")

    body = [
        _render_task_table(
            "Source task distribution",
            scorecard["source_task_counts"],
            scorecard["source_task_pct"],
            all_canon,
        ),
        _render_task_table(
            "Target task distribution",
            scorecard["target_task_counts"],
            scorecard["target_task_pct"],
            all_canon,
        ),
    ]

    pos = scorecard["position_type_counts"]
    pos_pct = scorecard["position_type_pct"]
    pt_lines = ["### Position-type mix (informational)", "", "| Position | Count | Share |", "|---|---:|---:|"]
    for k in sorted(pos):
        pt_lines.append(f"| `{k}` | {pos[k]} | {_fmt_pct(pos_pct.get(k, 0.0))} |")
    body.append("\n".join(pt_lines) + "\n")

    ec = scorecard["episode_coverage"]
    body.append(
        "### Episode coverage (informational)\n\n"
        f"- n_episodes: {ec['n_episodes']}\n"
        f"- rows/episode: min={ec['min_rows_per_episode']} "
        f"median={ec['median_rows_per_episode']:.1f} "
        f"mean={ec['mean_rows_per_episode']:.2f} "
        f"max={ec['max_rows_per_episode']}\n"
    )

    sd = scorecard["step_distribution"]
    sd_total = sum(sd.values()) or 1
    sd_lines = ["### Step-index distribution (informational)", "", "| Bucket | Count | Share |", "|---|---:|---:|"]
    for k in ("early_0_25", "mid_25_50", "mid_50_75", "late_75_100", "unknown"):
        v = sd.get(k, 0)
        if v:
            sd_lines.append(f"| `{k}` | {v} | {_fmt_pct(v / sd_total)} |")
    body.append("\n".join(sd_lines) + "\n")

    body.append(_render_pair_matrix(scorecard["pair_matrix"], all_canon))

    return "\n".join(head) + "\n" + "\n".join(body)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pairs", required=True, help="Counterfactual pairs JSONL.")
    p.add_argument(
        "--gate", action="store_true",
        help="Exit code 1 when any gate criterion fails. Without this flag the "
             "audit always exits 0 (useful for scorecard generation in CI).",
    )
    p.add_argument(
        "--json-out", default=None,
        help="Optional explicit path for the .audit.json (default: "
             "<pairs>.audit.json).",
    )
    p.add_argument(
        "--md-out", default=None,
        help="Optional explicit path for the .audit.md (default: "
             "<pairs>.audit.md).",
    )
    p.add_argument("--log-level", default="WARNING")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        print(f"FAIL: pairs file not found: {pairs_path}")
        return 1

    rows = _read_pairs(pairs_path)
    scorecard = build_scorecard(rows)

    json_out = Path(args.json_out) if args.json_out else pairs_path.with_suffix(pairs_path.suffix + ".audit.json")
    md_out = Path(args.md_out) if args.md_out else pairs_path.with_suffix(pairs_path.suffix + ".audit.md")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(scorecard, indent=2))
    md_out.write_text(render_markdown(scorecard, pairs_path))

    if scorecard["gate_pass"]:
        print(f"PASS  ({pairs_path.name}, n={scorecard['n_rows']}, cf={scorecard['counterfactual_fraction']:.2f})")
        return 0
    msg = "; ".join(scorecard["gate_failures"])
    print(f"FAIL  ({pairs_path.name}, n={scorecard['n_rows']}): {msg}")
    return 1 if args.gate else 0


if __name__ == "__main__":
    sys.exit(main())
