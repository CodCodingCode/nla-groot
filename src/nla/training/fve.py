"""Fraction of Variance Explained — the central reconstruction metric.

We compute FVE per-position (each example is already a single token position
in our setup, so per-example == per-position). The reference variance is the
batch mean activation; FVE measures how much the AR reconstruction beats
that baseline.

    FVE = 1 - sum((y - y_hat)^2) / sum((y - y_bar)^2)

where y_bar is the per-dim mean over the batch.

We also report cosine similarity since FVE alone can hide direction errors
when norms differ.

Stratified variants
-------------------
``StratifiedFve`` wraps multiple ``_StreamingFve`` instances keyed by an
arbitrary string (typically ``position_type`` -- ``last_text`` vs
``image_patch`` vs ``anchor``).  This is the metric that distinguishes the
backbone-image-position regime (where NLAs are uniquely valuable; SAE
features have no native readout there) from the language-position regime
(where the AV may degenerate into paraphrasing the instruction).
"""

from __future__ import annotations

from typing import Iterable

import torch


def fve_per_token(target: torch.Tensor, pred: torch.Tensor) -> dict[str, float]:
    """Compute FVE, MSE, cosine over a [B, H] batch.

    Args:
        target: ground-truth activations, shape [B, H].
        pred:   reconstructed activations, shape [B, H].
    """
    assert target.shape == pred.shape, f"shape mismatch: {target.shape} vs {pred.shape}"
    target = target.float()
    pred = pred.float()
    mean = target.mean(dim=0, keepdim=True)
    ss_res = ((target - pred) ** 2).sum().item()
    ss_tot = ((target - mean) ** 2).sum().item()
    if ss_tot == 0.0:
        return {"fve": float("nan"), "mse": ss_res / max(1, target.numel()), "cosine": 0.0}
    fve = 1.0 - ss_res / ss_tot
    mse = ss_res / target.numel()
    cos = torch.nn.functional.cosine_similarity(target, pred, dim=-1).mean().item()
    return {"fve": fve, "mse": mse, "cosine": cos}


def fve_streaming_accumulator():
    """Returns a small callable accumulator that maintains running FVE statistics.

    Usage:
        acc = fve_streaming_accumulator()
        acc.update(t, p)
        ...
        m = acc.compute()
    """
    return _StreamingFve()


class _StreamingFve:
    def __init__(self):
        self.n_elements = 0
        self.n_rows = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.ss_res = 0.0
        self.cos_sum = 0.0

    def update(self, target: torch.Tensor, pred: torch.Tensor) -> None:
        target = target.detach().float().cpu()
        pred = pred.detach().float().cpu()
        # Track element-wise sums (for FVE) and per-row cosine.
        self.n_elements += target.numel()
        self.n_rows += target.shape[0]
        self.sum += target.sum().item()
        self.sum_sq += (target ** 2).sum().item()
        self.ss_res += ((target - pred) ** 2).sum().item()
        self.cos_sum += torch.nn.functional.cosine_similarity(
            target, pred, dim=-1
        ).sum().item()

    def compute(self) -> dict[str, float]:
        if self.n_elements == 0:
            return {"fve": float("nan"), "mse": float("nan"), "cosine": float("nan")}
        mean = self.sum / self.n_elements
        ss_tot = self.sum_sq - self.n_elements * mean * mean
        cosine = self.cos_sum / max(1, self.n_rows)
        if ss_tot <= 0:
            return {"fve": float("nan"), "mse": self.ss_res / self.n_elements, "cosine": cosine}
        return {
            "fve": 1.0 - self.ss_res / ss_tot,
            "mse": self.ss_res / self.n_elements,
            "cosine": cosine,
        }


class StratifiedFve:
    """Maintain a separate ``_StreamingFve`` for each stratum key.

    ``update`` accepts a parallel list of strata keys (one per row of the
    target/pred batch) and routes each row to its own accumulator.  ``compute``
    returns ``{"all": metrics, "by_<group>/<key>": metrics, ...}`` so a caller
    can log scalars without juggling nested dicts.
    """

    def __init__(self, group_name: str = "position") -> None:
        self.group_name = group_name
        self._all = _StreamingFve()
        self._by_key: dict[str, _StreamingFve] = {}

    def update(
        self,
        target: torch.Tensor,
        pred: torch.Tensor,
        strata: Iterable[str],
    ) -> None:
        strata_list = list(strata)
        if target.shape[0] != len(strata_list):
            raise ValueError(
                f"strata length {len(strata_list)} != batch size {target.shape[0]}"
            )
        self._all.update(target, pred)
        # Bucket rows by stratum and update per-bucket accumulators in one pass.
        buckets: dict[str, list[int]] = {}
        for i, k in enumerate(strata_list):
            buckets.setdefault(str(k), []).append(i)
        for k, rows in buckets.items():
            idx = torch.tensor(rows, dtype=torch.long, device=target.device)
            self._by_key.setdefault(k, _StreamingFve()).update(
                target.index_select(0, idx), pred.index_select(0, idx),
            )

    def compute(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, v in self._all.compute().items():
            out[k] = v
        for stratum, acc in self._by_key.items():
            for k, v in acc.compute().items():
                out[f"{k}/{self.group_name}={stratum}"] = v
        return out
