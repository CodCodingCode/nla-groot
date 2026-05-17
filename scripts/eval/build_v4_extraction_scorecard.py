#!/usr/bin/env python
"""Aggregate the V4 extraction sweep into a single ranked scorecard.

Inputs are one ``data/sft/libero_4suite_v3/extraction_sweep/<layer>__<strategy>.json``
per config (produced by ``scripts/eval/probe_extraction_sweep.py``). This
script:

1. Loads every config row.
2. Computes the composite rank score (transparent weights, no hidden
   knobs), exactly as specified in the V4 plan::

       rank_score = 0.4 * knn_caption_at1
                  + 0.3 * suite_probe_acc
                  + 0.2 * same_ep_cosine_gap
                  - 0.1 * anisotropy_floor

3. Optionally bootstraps 95% confidence intervals on each metric and the
   composite by resampling configs ``--bootstrap-resamples`` times — when
   per-sample metric arrays aren't available (the proxy script stores
   only summary scalars), the bootstrap operates on a small Gaussian
   parametric perturbation of the reported value using a fixed sigma of
   ``0.02 * |value|``. That is intentionally a *coarse* uncertainty
   estimate; we use it only to flag "winner's lead is within noise"
   cases, never as a formal statistical test.
4. Writes ``v4_extraction_scorecard.json`` (machine-readable) and prints a
   human-readable ranked table to stdout.
5. Applies the decision criteria from the plan and emits the
   ``recommendation`` field:

      * winner's ``knn_caption_at1`` is >= 1.5x the random_one baseline
        in the same layer AND its 95% CI does not overlap with
        random_one's CI => ``ship``.
      * otherwise => ``a/b_top_two`` and the top two configs are listed
        for a Phase 2 short proxy-SFT.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/eval/build_v4_extraction_scorecard.py \\
        --sweep-root data/sft/libero_4suite_v3/extraction_sweep \\
        --diag-json  data/sft/libero_4suite_v3/extraction_diag.json \\
        --out-json   data/sft/libero_4suite_v3/v4_extraction_scorecard.json \\
        --bootstrap-resamples 100
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

logger = logging.getLogger("nla.v4_scorecard")


# Composite weights are spelled out as constants so anyone reading the
# scorecard JSON can immediately see how rank_score was computed.
COMPOSITE_WEIGHTS: dict[str, float] = {
    "knn_caption_at1":   +0.4,
    "suite_probe_acc":   +0.3,
    "same_ep_cosine_gap": +0.2,
    "anisotropy_floor":  -0.1,
}


def _config_key(row: dict) -> str:
    cfg = row.get("config", {})
    return f"L{cfg.get('layer', '?')}__{cfg.get('strategy', '?')}"


def _metric_value(row: dict, key: str) -> float:
    m = row.get("metrics") or {}
    entry = m.get(key) or {}
    # knn_*: "value" is the absolute Jaccard; "lift" is value - random_baseline.
    # For the composite we use lift on knn (more informative than absolute).
    if key.startswith("knn_caption_"):
        if "lift" in entry:
            return float(entry["lift"])
        return float(entry.get("value", 0.0))
    if key == "suite_probe_acc":
        return float(entry.get("value", 0.0))
    if key == "same_ep_cosine_gap":
        return float(entry.get("gap", 0.0))
    if key == "anisotropy_floor":
        return float(entry.get("value", 0.0))
    raise KeyError(f"Unknown metric key: {key}")


def composite_rank_score(row: dict) -> float:
    s = 0.0
    for key, w in COMPOSITE_WEIGHTS.items():
        s += w * _metric_value(row, key)
    return s


def bootstrap_ci(
    values: Iterable[float], *, n_resamples: int = 100, seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap of mean over a list of perturbed values."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size < 2:
        v = float(arr[0]) if arr.size else float("nan")
        return v, v
    means = []
    for _ in range(n_resamples):
        idx = rng.integers(0, arr.size, size=arr.size)
        means.append(arr[idx].mean())
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _parametric_perturbations(
    value: float, *, n: int, sigma_frac: float = 0.02, seed: int,
) -> list[float]:
    """Cheap CI proxy: Gaussian jitter around the reported value.

    Used because the proxy-eval script only stores summary scalars, not
    per-sample arrays. The CIs are therefore not statistically valid; they
    indicate "tight" vs "wide" lead. The plan explicitly accepts this
    trade-off ("we use it only to detect 'not enough signal to choose'").
    """
    rng = np.random.default_rng(seed)
    sigma = abs(value) * sigma_frac if value != 0 else sigma_frac
    return rng.normal(loc=value, scale=sigma, size=n).tolist()


def build_scorecard(
    sweep_root: Path,
    diag_json: Path | None,
    *,
    bootstrap_resamples: int = 100,
    seed: int = 0,
) -> dict:
    rows: list[dict] = []
    for p in sorted(sweep_root.glob("*.json")):
        if p.name == "v4_extraction_scorecard.json":
            continue
        try:
            rows.append(json.loads(p.read_text()))
        except Exception as e:
            logger.warning("Failed to parse %s: %s", p, e)
    if not rows:
        raise FileNotFoundError(f"No sweep rows under {sweep_root}")

    # Score every row, attach CIs.
    scored: list[dict] = []
    for row in rows:
        if "error" in row:
            logger.warning("config %s has error: %s", _config_key(row), row["error"])
            continue
        v = composite_rank_score(row)
        # CI on composite via parametric perturbation of each metric, recomposing.
        if bootstrap_resamples > 0:
            perturbed_composites: list[float] = []
            seed_base = seed + hash(_config_key(row)) % 2**31
            metric_perturbs: dict[str, list[float]] = {}
            for k in COMPOSITE_WEIGHTS:
                mv = _metric_value(row, k)
                metric_perturbs[k] = _parametric_perturbations(
                    mv, n=bootstrap_resamples,
                    seed=(seed_base + sum(ord(c) for c in k)),
                )
            for i in range(bootstrap_resamples):
                pc = sum(
                    COMPOSITE_WEIGHTS[k] * metric_perturbs[k][i]
                    for k in COMPOSITE_WEIGHTS
                )
                perturbed_composites.append(pc)
            ci_lo, ci_hi = np.percentile(perturbed_composites, [2.5, 97.5])
        else:
            ci_lo, ci_hi = v, v
        scored.append({
            "config":   row.get("config", {}),
            "config_key": _config_key(row),
            "n_samples": int(row.get("n_samples", 0)),
            "metrics_value": {
                k: _metric_value(row, k) for k in COMPOSITE_WEIGHTS
            },
            "rank_score": float(v),
            "rank_score_ci_lo": float(ci_lo),
            "rank_score_ci_hi": float(ci_hi),
            "metrics_raw": row.get("metrics", {}),
        })

    scored.sort(key=lambda r: -r["rank_score"])

    # Decision criteria.
    decision = _apply_decision(scored)

    out: dict = {
        "stage": "3_scorecard",
        "composite_weights": COMPOSITE_WEIGHTS,
        "bootstrap_resamples": int(bootstrap_resamples),
        "n_configs": len(scored),
        "rankings": scored,
        "decision": decision,
    }
    if diag_json and diag_json.exists():
        try:
            out["diag"] = json.loads(diag_json.read_text())
        except Exception:
            pass
    return out


def _apply_decision(rows: list[dict]) -> dict:
    if len(rows) < 2:
        return {"recommendation": "insufficient_configs", "n": len(rows)}
    winner = rows[0]
    runner = rows[1]

    # Decision criterion: winner's knn_caption_at1 must be >= 1.5x the
    # random_one-at-same-layer baseline AND its rank_score CI must not
    # overlap with random_one's.
    L = winner["config"].get("layer")
    rand_one_row = next(
        (r for r in rows
         if r["config"].get("layer") == L
         and r["config"].get("strategy") == "random_one"),
        None,
    )
    knn_winner = winner["metrics_raw"].get("knn_caption_at1", {}).get("value", 0.0)
    rand_one_knn = (rand_one_row or {}).get("metrics_raw", {}).get("knn_caption_at1", {}).get("value", 0.0)

    knn_ratio = knn_winner / rand_one_knn if rand_one_knn > 0 else float("inf")
    rand_ci_hi = (rand_one_row or {}).get("rank_score_ci_hi", -float("inf"))
    win_ci_lo = winner.get("rank_score_ci_lo", float("inf"))
    ci_disjoint = bool(win_ci_lo > rand_ci_hi)

    ship = bool(knn_ratio >= 1.5 and ci_disjoint)
    return {
        "winner": {
            "layer": winner["config"].get("layer"),
            "strategy": winner["config"].get("strategy"),
            "rank_score": winner["rank_score"],
            "rank_score_ci_lo": winner["rank_score_ci_lo"],
            "rank_score_ci_hi": winner["rank_score_ci_hi"],
            "knn_caption_at1": knn_winner,
        },
        "runner_up": {
            "layer": runner["config"].get("layer"),
            "strategy": runner["config"].get("strategy"),
            "rank_score": runner["rank_score"],
            "rank_score_ci_lo": runner["rank_score_ci_lo"],
            "rank_score_ci_hi": runner["rank_score_ci_hi"],
        },
        "random_one_baseline_at_winner_layer": {
            "layer": L,
            "strategy": "random_one",
            "knn_caption_at1": rand_one_knn,
        },
        "knn_ratio_winner_over_random_one": float(knn_ratio),
        "ci_disjoint_vs_random_one": ci_disjoint,
        "recommendation": "ship" if ship else "a_b_top_two",
        "reason": (
            "winner's knn@1 lift >= 1.5x random_one AND rank_score CIs disjoint"
            if ship
            else "lift below 1.5x or CIs overlap; recommend a short proxy-SFT A/B between top 2"
        ),
    }


def render_table(scored: list[dict]) -> str:
    """ASCII table for the console."""
    lines = []
    header = (
        f"{'rank':>4}  {'layer':>5}  {'strategy':<18}  "
        f"{'rank_score':>10}  {'CI_95':>17}  "
        f"{'knn@1':>7}  {'suite':>7}  {'gap':>7}  {'aniso':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for i, r in enumerate(scored, 1):
        cfg = r["config"]
        mv = r["metrics_value"]
        ci = f"[{r['rank_score_ci_lo']:+.3f},{r['rank_score_ci_hi']:+.3f}]"
        lines.append(
            f"{i:>4}  {cfg.get('layer', '?'):>5}  {cfg.get('strategy', '?'):<18}  "
            f"{r['rank_score']:>+10.4f}  {ci:>17}  "
            f"{mv['knn_caption_at1']:>+7.3f}  "
            f"{mv['suite_probe_acc']:>7.3f}  "
            f"{mv['same_ep_cosine_gap']:>+7.3f}  "
            f"{mv['anisotropy_floor']:>7.3f}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    REPO = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep-root", type=Path,
                   default=REPO / "data/sft/libero_4suite_v3/extraction_sweep")
    p.add_argument("--diag-json", type=Path,
                   default=REPO / "data/sft/libero_4suite_v3/extraction_diag.json")
    p.add_argument("--out-json", type=Path,
                   default=REPO / "data/sft/libero_4suite_v3/v4_extraction_scorecard.json")
    p.add_argument("--bootstrap-resamples", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    sc = build_scorecard(
        sweep_root=args.sweep_root,
        diag_json=args.diag_json,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(sc, indent=2) + "\n")
    logger.info("wrote %s", args.out_json)

    print("\n" + render_table(sc["rankings"]) + "\n")
    dec = sc["decision"]
    print(
        f"WINNER: layer={dec['winner']['layer']} strategy={dec['winner']['strategy']} "
        f"rank_score={dec['winner']['rank_score']:+.3f}"
    )
    print(
        f"runner-up: layer={dec['runner_up']['layer']} strategy={dec['runner_up']['strategy']} "
        f"rank_score={dec['runner_up']['rank_score']:+.3f}"
    )
    print(f"recommendation: {dec['recommendation']} ({dec['reason']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
