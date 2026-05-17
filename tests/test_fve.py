"""Tests for ``nla.training.fve``.

These tests pin the per-dimension batch-mean definition of FVE:
``_StreamingFve`` must agree with a single-shot ``fve_per_token`` call on the
concatenated batch, regardless of chunking. Prior to 2026-05 the streaming
implementation used a single scalar grand-mean baseline, which silently
disagreed with the docstring; these tests fence that off going forward.

See ``docs/sft_plan/v4_training_recon_audit.md`` section 4.1 for the algebra.
"""

from __future__ import annotations

import math

import pytest
import torch

from nla.training.fve import (
    StratifiedFve,
    _StreamingFve,
    fve_per_token,
    fve_streaming_accumulator,
)


def _make_batch(seed: int, B: int = 32, H: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Realistic-ish target/pred pair: pred = target + noise."""
    g = torch.Generator().manual_seed(seed)
    target = torch.randn(B, H, generator=g) * 5.0 + 1.5
    pred = target + 0.25 * torch.randn(B, H, generator=g)
    return target, pred


def test_streaming_matches_batch_on_single_update():
    """One chunk: streaming and batch must produce the same FVE."""
    target, pred = _make_batch(seed=0)
    batch = fve_per_token(target, pred)
    acc = _StreamingFve()
    acc.update(target, pred)
    streamed = acc.compute()
    assert math.isclose(streamed["fve"], batch["fve"], rel_tol=1e-6, abs_tol=1e-9)
    assert math.isclose(streamed["mse"], batch["mse"], rel_tol=1e-6, abs_tol=1e-9)
    assert math.isclose(streamed["cosine"], batch["cosine"], rel_tol=1e-6, abs_tol=1e-9)


@pytest.mark.parametrize("chunk_size", [1, 4, 7, 16])
def test_streaming_chunking_invariance(chunk_size: int):
    """Splitting a fixed batch into arbitrary chunks must not change FVE."""
    target, pred = _make_batch(seed=1, B=32, H=48)
    expected = fve_per_token(target, pred)
    acc = _StreamingFve()
    for start in range(0, target.shape[0], chunk_size):
        acc.update(target[start : start + chunk_size], pred[start : start + chunk_size])
    got = acc.compute()
    assert math.isclose(got["fve"], expected["fve"], rel_tol=1e-6, abs_tol=1e-9), (
        f"chunk_size={chunk_size}: streaming fve {got['fve']} != batch {expected['fve']}"
    )
    assert math.isclose(got["mse"], expected["mse"], rel_tol=1e-6, abs_tol=1e-9)
    assert math.isclose(got["cosine"], expected["cosine"], rel_tol=1e-6, abs_tol=1e-9)


def test_perfect_reconstruction_gives_fve_one():
    target, _ = _make_batch(seed=2)
    pred = target.clone()
    batch = fve_per_token(target, pred)
    acc = _StreamingFve()
    acc.update(target, pred)
    streamed = acc.compute()
    assert math.isclose(batch["fve"], 1.0, abs_tol=1e-9)
    assert math.isclose(streamed["fve"], 1.0, abs_tol=1e-9)
    assert math.isclose(batch["mse"], 0.0, abs_tol=1e-12)
    assert math.isclose(streamed["mse"], 0.0, abs_tol=1e-12)
    assert math.isclose(batch["cosine"], 1.0, rel_tol=1e-6)
    assert math.isclose(streamed["cosine"], 1.0, rel_tol=1e-6)


def test_predicting_mean_gives_fve_zero():
    """Predicting the per-dim mean of target reproduces the FVE baseline (=0)."""
    target, _ = _make_batch(seed=3, B=64, H=32)
    pred = target.mean(dim=0, keepdim=True).expand_as(target).clone()
    batch = fve_per_token(target, pred)
    acc = _StreamingFve()
    acc.update(target, pred)
    streamed = acc.compute()
    assert math.isclose(batch["fve"], 0.0, abs_tol=1e-6)
    assert math.isclose(streamed["fve"], 0.0, abs_tol=1e-6)


def test_streaming_is_not_global_mean_baseline():
    """Sanity check: the buggy global-mean baseline would inflate ss_tot.

    Construct a target with strongly differing per-dim means; the per-dim
    baseline ss_tot is ~0 (since each dim is nearly constant) while the
    global-mean baseline is large. With perfect prediction both definitions
    give fve=1 (trivial), so we instead set pred = a constant scalar offset
    and check that ``_StreamingFve`` agrees with ``fve_per_token`` (per-dim)
    -- and disagrees with the old global-mean definition.
    """
    H = 8
    # Per-dim means at very different scales; within-dim noise is tiny.
    target = torch.stack(
        [torch.full((50,), float(d * 100)) + 0.01 * torch.randn(50) for d in range(H)],
        dim=1,
    )
    pred = target + 0.5  # constant offset error
    batch = fve_per_token(target, pred)
    acc = _StreamingFve()
    acc.update(target, pred)
    streamed = acc.compute()
    assert math.isclose(streamed["fve"], batch["fve"], rel_tol=1e-3, abs_tol=1e-6)

    # The old (buggy) global-scalar-mean denominator -- compute by hand and
    # confirm it gives a *materially different* FVE. This is the "wrong" path.
    grand_mean = target.mean()
    ss_tot_old = ((target - grand_mean) ** 2).sum().item()
    ss_res = ((target - pred) ** 2).sum().item()
    fve_old = 1.0 - ss_res / ss_tot_old
    assert not math.isclose(fve_old, batch["fve"], rel_tol=0.01, abs_tol=0.05), (
        "global-mean baseline should disagree with per-dim baseline on this "
        f"contrived input (per-dim={batch['fve']}, global={fve_old})"
    )


def test_streaming_accumulator_factory():
    """Public ``fve_streaming_accumulator`` returns a working _StreamingFve."""
    target, pred = _make_batch(seed=5)
    acc = fve_streaming_accumulator()
    acc.update(target, pred)
    result = acc.compute()
    assert "fve" in result and "mse" in result and "cosine" in result


def test_streaming_rejects_dim_mismatch():
    acc = _StreamingFve()
    acc.update(torch.randn(4, 16), torch.randn(4, 16))
    with pytest.raises(ValueError, match="hidden dim"):
        acc.update(torch.randn(4, 32), torch.randn(4, 32))


def test_streaming_rejects_shape_mismatch():
    acc = _StreamingFve()
    with pytest.raises(ValueError, match="shape mismatch"):
        acc.update(torch.randn(4, 16), torch.randn(4, 17))


def test_streaming_rejects_non_2d():
    acc = _StreamingFve()
    with pytest.raises(ValueError, match=r"\[B, H\]"):
        acc.update(torch.randn(4, 8, 16), torch.randn(4, 8, 16))


def test_empty_returns_nan():
    acc = _StreamingFve()
    result = acc.compute()
    assert math.isnan(result["fve"])
    assert math.isnan(result["mse"])
    assert math.isnan(result["cosine"])


def test_stratified_fve_matches_per_stratum_batch():
    """StratifiedFve per-stratum buckets must agree with per-stratum batch FVE."""
    target_a, pred_a = _make_batch(seed=10, B=20, H=16)
    target_b, pred_b = _make_batch(seed=11, B=30, H=16)
    # Interleave two strata in a single update.
    target = torch.cat([target_a, target_b], dim=0)
    pred = torch.cat([pred_a, pred_b], dim=0)
    strata = ["a"] * target_a.shape[0] + ["b"] * target_b.shape[0]
    s = StratifiedFve(group_name="kind")
    s.update(target, pred, strata)
    out = s.compute()

    # Overall (mixed) FVE comes from the underlying _all accumulator.
    expected_all = fve_per_token(target, pred)
    assert math.isclose(out["fve"], expected_all["fve"], rel_tol=1e-6, abs_tol=1e-9)

    # Per-stratum FVE matches a direct batch call on the same slice.
    expected_a = fve_per_token(target_a, pred_a)
    expected_b = fve_per_token(target_b, pred_b)
    assert math.isclose(out["fve/kind=a"], expected_a["fve"], rel_tol=1e-6, abs_tol=1e-9)
    assert math.isclose(out["fve/kind=b"], expected_b["fve"], rel_tol=1e-6, abs_tol=1e-9)


def test_stratified_fve_chunking_invariance():
    """Splitting StratifiedFve.update across calls must not change results."""
    target, _ = _make_batch(seed=20, B=40, H=24)
    pred = target + 0.1 * torch.randn(40, 24, generator=torch.Generator().manual_seed(21))
    strata = ["a", "b", "c"] * 13 + ["a"]
    assert len(strata) == 40

    s1 = StratifiedFve(group_name="g")
    s1.update(target, pred, strata)
    out1 = s1.compute()

    s2 = StratifiedFve(group_name="g")
    for start in range(0, 40, 7):
        s2.update(
            target[start : start + 7],
            pred[start : start + 7],
            strata[start : start + 7],
        )
    out2 = s2.compute()

    for k in out1:
        assert math.isclose(out1[k], out2[k], rel_tol=1e-6, abs_tol=1e-9), (
            f"chunked StratifiedFve disagrees on key {k}: one-shot={out1[k]} chunked={out2[k]}"
        )
