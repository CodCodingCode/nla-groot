#!/usr/bin/env python
"""Diagnostic aggregator for ``sim_reward_cache.jsonl``.

This is a **diagnostic only** tool. The numbers it produces are not eligible
to be paper headlines because the cache mixes train activations, repeated
seeds, and stale entries from previous runs. Use it to:

  - Confirm sim infra is healthy (low error rate, non-zero predicate hits)
  - See which ``target_task``s are well-populated in the cache
  - Spot stratification problems (e.g. all goal predicate hits but no
    cross-suite ones)

For publishable claims always use ``compare_cf_steer_checkpoints.py`` on the
held-out eval slice, NOT this script.

Usage::

    PYTHONPATH=src python scripts/eval/aggregate_sim_cache.py \\
        --cache data/grpo/.../sim_reward_cache.jsonl \\
        --out-json data/grpo/.../sim_reward_cache_aggregate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


_BANNER = (
    "[diagnostic] sim_reward_cache aggregates are NOT publishable headlines. "
    "Use compare_cf_steer_checkpoints.py on a held-out CF eval slice."
)


def _iter_cache(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cache", required=True,
                   help="Path to sim_reward_cache.jsonl from a GRPO run.")
    p.add_argument("--out-json", required=True,
                   help="Path to write the aggregate JSON.")
    p.add_argument("--max-rows-by-task", type=int, default=20,
                   help="Cap the per-target_task table to N tasks (default 20).")
    args = p.parse_args(argv)

    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"FATAL: cache missing: {cache_path}", file=sys.stderr)
        return 2

    print(_BANNER, file=sys.stderr)

    n_rows = 0
    n_with_predicate = 0
    n_with_success_any = 0
    n_with_error = 0
    r_sim_sum = 0.0
    r_sim_count = 0
    by_task: dict[str, Counter] = {}
    by_env: dict[str, Counter] = {}
    unique_source_ids: set[str] = set()
    for row in _iter_cache(cache_path):
        n_rows += 1
        if row.get("error"):
            n_with_error += 1
        pred = _safe_float(row.get("predicate"))
        if pred is not None and pred > 0:
            n_with_predicate += 1
        if row.get("success_any"):
            n_with_success_any += 1
        rs = _safe_float(row.get("r_sim"))
        if rs is not None:
            r_sim_sum += rs
            r_sim_count += 1
        task = str(row.get("target_task") or "(unknown)")
        env = str(row.get("env_name") or "(unknown)")
        sid = str(row.get("source_id") or "")
        if sid:
            unique_source_ids.add(sid)
        t = by_task.setdefault(task, Counter())
        t["n"] += 1
        if pred is not None and pred > 0:
            t["predicate_pos"] += 1
        if row.get("success_any"):
            t["bddl_native_pos"] += 1
        e = by_env.setdefault(env, Counter())
        e["n"] += 1
        if pred is not None and pred > 0:
            e["predicate_pos"] += 1

    def _rate(c: Counter, num_key: str, denom_key: str = "n") -> float:
        n = int(c.get(denom_key, 0))
        if n == 0:
            return 0.0
        return float(c.get(num_key, 0)) / n

    per_task = {
        task: {
            "n": int(c.get("n", 0)),
            "predicate_rate_diagnostic": _rate(c, "predicate_pos"),
            "bddl_native_rate_diagnostic": _rate(c, "bddl_native_pos"),
        }
        for task, c in sorted(by_task.items(), key=lambda kv: -kv[1].get("n", 0))
    }
    per_env = {
        env: {
            "n": int(c.get("n", 0)),
            "predicate_rate_diagnostic": _rate(c, "predicate_pos"),
        }
        for env, c in sorted(by_env.items(), key=lambda kv: -kv[1].get("n", 0))
    }

    aggregate: dict[str, Any] = {
        "schema_version": 1,
        "_warning":
            "diagnostic only -- not for paper headlines (use held-out compare)",
        "cache_path": str(cache_path),
        "n_rows": n_rows,
        "n_unique_source_ids": len(unique_source_ids),
        "n_with_predicate_pos": n_with_predicate,
        "n_with_success_any": n_with_success_any,
        "n_with_error": n_with_error,
        "predicate_rate_diagnostic": (
            n_with_predicate / n_rows if n_rows else 0.0
        ),
        "bddl_native_rate_diagnostic": (
            n_with_success_any / n_rows if n_rows else 0.0
        ),
        "error_rate_diagnostic": (
            n_with_error / n_rows if n_rows else 0.0
        ),
        "mean_r_sim_diagnostic": (
            r_sim_sum / r_sim_count if r_sim_count else 0.0
        ),
        "per_target_task": dict(list(per_task.items())[: args.max_rows_by_task]),
        "per_env_name": dict(list(per_env.items())[: args.max_rows_by_task]),
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(aggregate, indent=2))

    print(_BANNER)
    print(
        f"cache rows: {n_rows} (unique source_ids: {len(unique_source_ids)})"
    )
    print(
        f"  diagnostic predicate-positive: {n_with_predicate} "
        f"({aggregate['predicate_rate_diagnostic']:.2%})"
    )
    print(
        f"  diagnostic bddl-native-positive: {n_with_success_any} "
        f"({aggregate['bddl_native_rate_diagnostic']:.2%})"
    )
    print(
        f"  diagnostic error rate: {aggregate['error_rate_diagnostic']:.2%}"
    )
    print(
        f"  diagnostic mean r_sim: {aggregate['mean_r_sim_diagnostic']:.3f}"
    )
    print(f"  Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
