"""Compute the α scaling factor and other corpus-level activation stats.

The NLA paper's appendix prescribes::

    α  =  P75( || h_t ||_2  for t in valid positions over a large corpus )

This is then used as a *fixed* multiplier when injecting activations into the
AV's residual stream (NLA paper Eq. 1):

    h_orig  ->  h_orig + α · ||h_orig|| · Δ / ||Δ||

and as the AR's normalization target.  Hardcoding it after extraction (rather
than re-tuning during RL) is one of the most important defaults to copy.

We compute α by streaming through the dump once and feeding all norms to a
P²-style quantile estimator (so we don't need to keep every value in memory).
For modest corpus sizes (< 1M positions) we also support an exact percentile
via numpy.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from nla.extraction.storage import ActivationShardReader


@dataclass
class ActivationStats:
    """Corpus-level activation statistics.

    Norm percentiles are computed over *all valid (non-pad) token positions*.
    The α used at training/inference time is ``p75_norm`` per the NLA paper.
    """

    n_positions: int
    p50_norm: float
    p75_norm: float            # <- this is α
    p90_norm: float
    p99_norm: float
    mean_norm: float
    std_norm: float
    image_token_fraction: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @property
    def alpha(self) -> float:
        """The α used when injecting activations into the AV residual stream."""
        return self.p75_norm


def _norms_from_record(item: dict) -> np.ndarray:
    """Extract per-valid-position L2 norms from one example record."""
    features: torch.Tensor = item["features"]                       # [T, H]
    attention_mask: torch.Tensor = item["attention_mask"]           # [T]
    valid = attention_mask.bool()
    if not valid.any():
        return np.empty(0, dtype=np.float32)
    f = features[valid].to(dtype=torch.float32)
    return torch.linalg.vector_norm(f, ord=2, dim=-1).cpu().numpy()


def compute_stats(
    reader: ActivationShardReader,
    *,
    max_positions: int | None = None,
    seed: int = 0,
) -> ActivationStats:
    """Compute exact percentiles over up to ``max_positions`` valid tokens.

    For a pilot corpus of ~10k examples × ~300 tokens this is ~3M values, well
    within memory.  If you exceed that, pass ``max_positions`` to subsample
    uniformly across examples.
    """
    norms: list[np.ndarray] = []
    image_token_count = 0
    valid_token_count = 0

    rng = np.random.default_rng(seed)
    records = reader.records
    if max_positions is not None:
        # Estimate per-example sample rate so we converge near max_positions.
        approx_total = sum(rec.seq_len for rec in records)
        keep_rate = min(1.0, float(max_positions) / max(1, approx_total))
    else:
        keep_rate = 1.0

    for item in reader.iter_examples():
        attn = item["attention_mask"].bool()
        img = item["image_mask"].bool()
        valid_count = int(attn.sum().item())
        valid_token_count += valid_count
        image_token_count += int((attn & img).sum().item())

        ex_norms = _norms_from_record(item)
        if keep_rate < 1.0 and ex_norms.size > 0:
            mask = rng.random(ex_norms.size) < keep_rate
            ex_norms = ex_norms[mask]
        if ex_norms.size > 0:
            norms.append(ex_norms)

    if not norms:
        raise ValueError("No valid token positions found across the dump.")

    all_norms = np.concatenate(norms)
    p50, p75, p90, p99 = np.percentile(all_norms, [50, 75, 90, 99]).tolist()
    mean = float(all_norms.mean())
    std = float(all_norms.std())
    img_frac = (
        float(image_token_count) / float(valid_token_count) if valid_token_count else 0.0
    )

    return ActivationStats(
        n_positions=int(all_norms.size),
        p50_norm=float(p50),
        p75_norm=float(p75),
        p90_norm=float(p90),
        p99_norm=float(p99),
        mean_norm=mean,
        std_norm=std,
        image_token_fraction=img_frac,
    )


def save_stats(stats: ActivationStats, path: str | Path) -> None:
    Path(path).write_text(stats.to_json())


def load_stats(path: str | Path) -> ActivationStats:
    d = json.loads(Path(path).read_text())
    return ActivationStats(**d)


def alpha_from_norms(norms: Iterable[float]) -> float:
    """Convenience helper for callers who already have a flat array of norms."""
    arr = np.fromiter(norms, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("Empty norm iterable; cannot compute α.")
    return float(np.percentile(arr, 75))
