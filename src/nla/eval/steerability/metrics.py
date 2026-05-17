"""Aggregate per-episode summaries into per-condition tables + comparisons."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else None


def _frac(xs: list[bool]) -> float | None:
    if not xs:
        return None
    return float(np.mean([1.0 if x else 0.0 for x in xs]))


def aggregate_condition(
    summaries: list[dict[str, Any]],
    tracked_bodies: list[str],
) -> dict[str, Any]:
    """Roll up per-episode summaries into per-condition averages."""
    if not summaries:
        return {}
    out: dict[str, Any] = {
        "n_episodes": len(summaries),
        "success_any_rate": _frac([s.get("success_any", False) for s in summaries]),
        "success_final_rate": _frac([s.get("success_final", False) for s in summaries]),
        "mean_n_steps": _mean([s["n_steps"] for s in summaries]),
    }
    per_obj_disp: dict[str, list[float]] = defaultdict(list)
    per_obj_minee: dict[str, list[float]] = defaultdict(list)
    for s in summaries:
        for k, v in (s.get("displacement") or {}).items():
            per_obj_disp[k].append(v)
        for k, v in (s.get("min_ee_distance") or {}).items():
            per_obj_minee[k].append(v)
    out["displacement"] = {k: _mean(v) for k, v in per_obj_disp.items()}
    out["min_ee_distance"] = {k: _mean(v) for k, v in per_obj_minee.items()}
    # Winner distribution
    winners = [s.get("displacement_winner") for s in summaries]
    winner_counts: dict[str, int] = defaultdict(int)
    for w in winners:
        if w is not None:
            winner_counts[w] += 1
    out["winner_counts"] = dict(winner_counts)
    # Target-aware metrics if target was set on the condition (consistent across eps)
    target_disps = [s["target_displacement"] for s in summaries if "target_displacement" in s]
    target_minees = [s["target_min_ee_distance"] for s in summaries if "target_min_ee_distance" in s]
    target_wins = [s["target_winner"] for s in summaries if "target_winner" in s]
    if target_disps:
        out["mean_target_displacement"] = _mean(target_disps)
    if target_minees:
        out["mean_target_min_ee_distance"] = _mean(target_minees)
    if target_wins:
        out["target_winner_rate"] = _frac(target_wins)
    return out


def aggregate_all(
    output_dir: Path,
    condition_names: list[str],
    env_names: list[str],
    tracked_bodies: list[str],
) -> dict[str, Any]:
    """Walk the harness output tree and produce ``metrics.json``."""
    results: dict[str, Any] = {"conditions": {}, "envs": env_names}
    for cond in condition_names:
        cond_dir = output_dir / "conditions" / cond
        per_env: dict[str, Any] = {}
        all_eps: list[dict[str, Any]] = []
        for env in env_names:
            env_dir = cond_dir / env.replace("/", "__")
            ep_summaries: list[dict[str, Any]] = []
            for sd in sorted(env_dir.glob("seed_*")):
                sj = sd / "summary.json"
                if sj.exists():
                    ep_summaries.append(json.loads(sj.read_text()))
            per_env[env] = aggregate_condition(ep_summaries, tracked_bodies)
            all_eps.extend(ep_summaries)
        results["conditions"][cond] = {
            "by_env": per_env,
            "overall": aggregate_condition(all_eps, tracked_bodies),
        }
    return results
