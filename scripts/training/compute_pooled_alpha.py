#!/usr/bin/env python
"""Compute α (P75 ‖h‖₂) over **mean-pooled image_patch** vectors.

Why a new α
-----------
V3 stored α = P75 ‖h‖₂ over *every valid token* in the dump:

    data/activations/libero_4suite_v4_combined/stats.json
      p75_norm ≈ 204

That is roughly the norm of a single image-patch token. V4 SFT replaces
each ``image_patch`` row's target with the *mean of all image-patch
tokens* in that example (``image_patch_pooling=mean_pool_image`` in
``LabeledPositionDataset``). Mean-pooling shortens the vector by ~25%
(opposite-direction noise cancels in the average), so the V3 α is now
~25% too high for the ptype that supplies ~50% of the SFT batch.

Effect on training, if uncorrected:
  * AR's reconstruction target on image_patch rows is ``h/α``.
    With α too high, the target shrinks → AR is rewarded for predicting
    tiny vectors → AR under-predicts.
  * AR-MSE on image_patch rows is therefore inflated.
  * The relative AR-MSE vs AV-CE balance shifts; the model can adapt
    but burns optimizer steps on a calibration the dataset already
    knows.

What this script does
---------------------
1. Open the activations dump at ``--activations-root``.
2. Pick ``--n-samples`` examples (default 20k) uniformly at random.
3. For each example: ``mean_pool_image(features, image_mask,
   attention_mask)`` → a single ``[H]`` pooled vector.
4. Take its L2 norm.
5. Compute P50 / P75 / P90 / P99 / mean / std over the resulting norms.
6. Emit an ``ActivationStats``-compatible JSON at ``--output`` so
   ``run_sft.py --stats-json <output>`` picks up the new α with zero
   training-side changes.

Usage::

    PYTHONPATH=src python scripts/training/compute_pooled_alpha.py \\
        --activations-root data/activations/libero_4suite_v4_combined \\
        --output           data/activations/libero_4suite_v4_combined/stats_pooled.json \\
        --n-samples 20000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

# Allow ``python scripts/...`` invocation without an explicit PYTHONPATH=src.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from nla.extraction.position_strategies import mean_pool_image
from nla.extraction.stats import ActivationStats
from nla.extraction.storage import ActivationShardReader


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--activations-root", required=True,
                   help="Directory produced by run_extract.py (must contain "
                        "manifest.json, index.jsonl, shard_NNNNNN/).")
    p.add_argument("--output", required=True,
                   help="Where to write the pooled stats.json. ActivationStats "
                        "schema; pass directly to run_sft.py --stats-json.")
    p.add_argument("--n-samples", type=int, default=20000,
                   help="Number of pooled image_patch vectors to sample. "
                        "20k is enough for a stable P75 (~3 sf).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--compare-to",
                   default=None,
                   help="Optional path to an existing stats.json (e.g. V3 "
                        "per-token). If provided, the script prints a "
                        "side-by-side V3-vs-pooled summary before exiting.")
    return p


def _compute_pooled_norms(
    reader: ActivationShardReader,
    n_samples: int,
    seed: int,
    log: logging.Logger,
) -> np.ndarray:
    """Return ``n`` L2 norms of mean-pooled image_patch vectors.

    Samples ``n_samples`` example indices uniformly at random (without
    replacement when n <= corpus size) and groups them by shard to amortize
    safetensors open() across the iteration.
    """
    records = reader.records
    n_total = len(records)
    if n_total == 0:
        raise RuntimeError("Empty corpus: no records in the index.")
    rng = np.random.default_rng(seed)
    if n_samples >= n_total:
        log.warning(
            "Requested n_samples=%d >= corpus size %d; pooling every example.",
            n_samples, n_total,
        )
        chosen_idx = np.arange(n_total)
    else:
        chosen_idx = rng.choice(n_total, size=n_samples, replace=False)
    chosen_set = set(int(i) for i in chosen_idx)

    norms = np.empty(len(chosen_idx), dtype=np.float64)
    pos = 0
    skipped = 0
    t0 = time.time()
    last_log = t0

    chosen_ids = {records[int(i)].example_id for i in chosen_idx}

    for item in reader.iter_examples(
        record_filter=lambda rec: rec.example_id in chosen_ids
    ):
        try:
            pooled = mean_pool_image(
                features=item["features"].to(torch.float32),
                image_mask=item["image_mask"],
                attention_mask=item["attention_mask"],
            )
        except ValueError:
            skipped += 1
            continue
        n = float(torch.linalg.vector_norm(pooled, ord=2).item())
        norms[pos] = n
        pos += 1
        now = time.time()
        if now - last_log >= 5.0:
            log.info(
                "  ... %d / %d pooled vectors (%.0f/s, %d skipped)",
                pos, len(chosen_idx), pos / max(now - t0, 1e-6), skipped,
            )
            last_log = now

    if pos == 0:
        raise RuntimeError(
            "No pooled vectors produced — does the corpus actually have "
            "image-patch tokens? (Check image_mask in the dump.)"
        )
    if skipped:
        log.warning("Skipped %d examples with no image_patch tokens.", skipped)
    return norms[:pos]


def _stats_from_norms(norms: np.ndarray, image_token_fraction: float) -> ActivationStats:
    p50, p75, p90, p99 = np.percentile(norms, [50, 75, 90, 99]).tolist()
    return ActivationStats(
        n_positions=int(norms.size),
        p50_norm=float(p50),
        p75_norm=float(p75),
        p90_norm=float(p90),
        p99_norm=float(p99),
        mean_norm=float(norms.mean()),
        std_norm=float(norms.std()),
        image_token_fraction=float(image_token_fraction),
    )


def _print_comparison(
    pooled: ActivationStats,
    compare_path: Path,
    log: logging.Logger,
) -> None:
    d = json.loads(compare_path.read_text())
    log.info("Comparison vs %s:", compare_path)
    log.info("                       V3 per-token     V4 pooled       Δ%%")
    for k in ("p50_norm", "p75_norm", "p90_norm", "p99_norm", "mean_norm"):
        v3 = float(d[k])
        v4 = getattr(pooled, k)
        delta = 100.0 * (v4 - v3) / v3 if v3 else 0.0
        log.info("  %-18s   %12.3f   %12.3f   %+6.1f%%", k, v3, v4, delta)
    log.info("  α (P75)            % 12.3f → %12.3f   %+6.1f%%",
             float(d["p75_norm"]),
             pooled.p75_norm,
             100.0 * (pooled.p75_norm - float(d["p75_norm"])) / float(d["p75_norm"]),
             )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    log = logging.getLogger("compute_pooled_alpha")

    log.info("Loading dump %s ...", args.activations_root)
    reader = ActivationShardReader(args.activations_root)
    log.info("  corpus size: %d examples", len(reader.records))
    log.info("  n_samples:   %d", args.n_samples)

    norms = _compute_pooled_norms(
        reader, n_samples=args.n_samples, seed=args.seed, log=log,
    )
    log.info("Computed %d pooled-image_patch norms.", norms.size)

    # image_token_fraction is meaningless once we pool to a single vector
    # per example, but ActivationStats requires it. Set to 1.0 — every
    # sampled vector IS an image-patch summary.
    stats = _stats_from_norms(norms, image_token_fraction=1.0)
    log.info(
        "Stats: p50=%.3f p75=%.3f (α) p90=%.3f p99=%.3f mean=%.3f std=%.3f",
        stats.p50_norm, stats.p75_norm, stats.p90_norm,
        stats.p99_norm, stats.mean_norm, stats.std_norm,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(stats), indent=2))
    log.info("Wrote %s", out_path)

    if args.compare_to:
        _print_comparison(stats, Path(args.compare_to), log)

    log.info(
        "Done. Pass --stats-json %s to scripts/training/run_sft.py to wire "
        "the new α into AV/AR for V4 training.",
        out_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
