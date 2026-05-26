"""Fraction of Variance Explained -- the central reconstruction metric.

We compute FVE per-position (each example is already a single token position
in our setup, so per-example == per-position). The reference variance is the
per-dimension batch mean of ``target``; FVE measures how much the AR
reconstruction beats that baseline.

    FVE = 1 - sum((y - y_hat)^2) / sum((y - y_bar_d)^2)

where ``y_bar_d`` is the per-dimension mean of ``target`` over the batch
(broadcast back to the original shape). ``fve_per_token`` computes this
directly on a single batch; ``_StreamingFve`` accumulates the same statistics
across batches so the value of ``fve`` produced by ``compute()`` is identical
(up to floating-point) to running ``fve_per_token`` once on the concatenated
batch.

We also report cosine similarity since FVE alone can hide direction errors
when norms differ.

Stratified variants
-------------------
``StratifiedFve`` wraps multiple ``_StreamingFve`` instances keyed by an
arbitrary string (typically ``position_type`` -- ``last_text`` vs
``image_patch`` vs ``anchor``). This is the metric that distinguishes the
backbone-image-position regime (where NLAs are uniquely valuable; SAE
features have no native readout there) from the language-position regime
(where the AV may degenerate into paraphrasing the instruction).

History / metric-definition note
--------------------------------
Before 2026-05 ``_StreamingFve`` accumulated a single scalar grand-mean over
all elements (batch x hidden) and used that as the baseline for ``ss_tot``.
That denominator is algebraically always at least as large as the per-dim
denominator above, so the logged ``fve`` was optimistically biased vs the
docstring definition. Runs evaluated before the fix (e.g. the V3 SFT under
``data/sft/libero_4suite_v3/``) used the global-mean baseline; their ``fve``
numbers are therefore not directly comparable to runs evaluated after this
file changes -- ``mse`` and ``cosine`` are unaffected. See
``docs/sft_plan/v4_training_recon_audit.md`` section 4.1 for the derivation.
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
    """Online accumulator equivalent to ``fve_per_token`` on the concatenation.

    Maintains per-dimension running ``sum`` and ``sum of squares`` of
    ``target`` so the FVE baseline at ``compute()`` time is
    ``y_bar_d = sum_per_dim / n_rows``, matching the module docstring's
    per-dimension batch-mean definition. ``ss_res`` is accumulated element-wise
    as before.
    """

    def __init__(self):
        self.n_rows: int = 0
        self.n_elements: int = 0
        self.ss_res: float = 0.0
        self.cos_sum: float = 0.0
        # Allocated lazily on first update so we don't pin a fixed H here.
        self._sum_per_dim: torch.Tensor | None = None
        self._sum_sq_per_dim: torch.Tensor | None = None

    def _ensure_buffers(self, hidden: int) -> None:
        if self._sum_per_dim is None:
            self._sum_per_dim = torch.zeros(hidden, dtype=torch.float64)
            self._sum_sq_per_dim = torch.zeros(hidden, dtype=torch.float64)
        elif self._sum_per_dim.shape[0] != hidden:
            raise ValueError(
                f"_StreamingFve received target with hidden dim {hidden} but "
                f"was previously updated with hidden dim {self._sum_per_dim.shape[0]}"
            )

    def update(self, target: torch.Tensor, pred: torch.Tensor) -> None:
        if target.shape != pred.shape:
            raise ValueError(
                f"_StreamingFve shape mismatch: target {tuple(target.shape)} vs pred {tuple(pred.shape)}"
            )
        if target.dim() != 2:
            raise ValueError(
                f"_StreamingFve expects [B, H] tensors; got {tuple(target.shape)}"
            )
        target = target.detach().float().cpu()
        pred = pred.detach().float().cpu()
        B, H = target.shape
        self._ensure_buffers(H)
        self.n_rows += B
        self.n_elements += target.numel()
        # fp64 buffers avoid catastrophic cancellation in ss_tot for large
        # corpora; cast to float64 here so the running sums are exact-ish.
        t64 = target.to(torch.float64)
        self._sum_per_dim += t64.sum(dim=0)
        self._sum_sq_per_dim += (t64 * t64).sum(dim=0)
        self.ss_res += ((target - pred) ** 2).sum().item()
        self.cos_sum += torch.nn.functional.cosine_similarity(
            target, pred, dim=-1
        ).sum().item()

    def compute(self) -> dict[str, float]:
        if self.n_rows == 0 or self._sum_per_dim is None:
            return {"fve": float("nan"), "mse": float("nan"), "cosine": float("nan")}
        cosine = self.cos_sum / max(1, self.n_rows)
        mse = self.ss_res / self.n_elements
        # Per-dim variance summed over hidden dims:
        #   ss_tot = sum_d ( sum_sq_d - sum_d^2 / n_rows )
        ss_tot_per_dim = self._sum_sq_per_dim - (self._sum_per_dim ** 2) / float(self.n_rows)
        ss_tot = float(ss_tot_per_dim.sum().item())
        if ss_tot <= 0.0:
            return {"fve": float("nan"), "mse": mse, "cosine": cosine}
        return {
            "fve": 1.0 - self.ss_res / ss_tot,
            "mse": mse,
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
        # v7 spatial AR: target/pred can be ``(B, K, H)``. Flatten the K
        # axis into the batch dim and repeat each stratum K times so the
        # underlying ``_StreamingFve`` accumulates the per-position residuals
        # as independent observations. This preserves the "fraction of
        # variance explained" semantics: each (sample, position) is a draw
        # from the empirical distribution of image-patch activations.
        if target.dim() == 3 and pred.dim() == 3:
            B, K = target.shape[0], target.shape[1]
            target = target.reshape(B * K, target.shape[-1])
            pred = pred.reshape(B * K, pred.shape[-1])
            strata_list = [s for s in strata_list for _ in range(K)]
        elif target.dim() != 2 or pred.dim() != 2:
            raise ValueError(
                f"StratifiedFve expects (B, H) or (B, K, H); got target "
                f"{tuple(target.shape)} pred {tuple(pred.shape)}"
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
