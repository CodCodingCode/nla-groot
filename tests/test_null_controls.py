"""Tests for the matched-null and shuffled non-semantic control vectors.

These vectors are used by the closed-loop CF compare to argue that any
predicate-rate gain is **semantic**, not just norm injection. The contract
is: same L2 magnitude as the real ``AR(text)`` output, but no semantic
content. Tests pin determinism + magnitude preservation.
"""

from __future__ import annotations

import pytest
import torch

from nla.steering.null_controls import matched_null_vec, shuffled_vec


def test_matched_null_preserves_l2_norm() -> None:
    real = torch.tensor([3.0, 4.0, 0.0])
    target_norm = float(torch.linalg.norm(real))
    out = matched_null_vec(real, seed=42)
    assert out.shape == real.shape
    assert float(torch.linalg.norm(out)) == pytest.approx(target_norm, rel=1e-6)
    # Should be float32 CPU regardless of input dtype.
    assert out.dtype == torch.float32
    assert out.device.type == "cpu"


def test_matched_null_deterministic_per_seed() -> None:
    real = torch.ones(64)
    a = matched_null_vec(real, seed=7)
    b = matched_null_vec(real, seed=7)
    c = matched_null_vec(real, seed=8)
    assert torch.allclose(a, b)
    assert not torch.allclose(a, c)


def test_matched_null_independent_of_real_direction() -> None:
    """Direction comes from the RNG; ``real`` only sets the norm."""
    real_a = torch.tensor([1.0, 0.0, 0.0])
    real_b = torch.tensor([0.0, 0.0, 1.0])  # same norm, different direction
    out_a = matched_null_vec(real_a, seed=3)
    out_b = matched_null_vec(real_b, seed=3)
    # Same RNG seed → same direction; norms match each input's norm.
    assert torch.allclose(out_a, out_b)


def test_matched_null_zero_input_returns_zero() -> None:
    real = torch.zeros(8)
    out = matched_null_vec(real, seed=1)
    assert float(torch.linalg.norm(out)) == pytest.approx(0.0)


def test_shuffled_vec_preserves_norm_and_multiset() -> None:
    real = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = shuffled_vec(real, seed=11)
    assert out.shape == real.shape
    assert float(torch.linalg.norm(out)) == pytest.approx(
        float(torch.linalg.norm(real)), rel=1e-6
    )
    assert sorted(out.tolist()) == sorted(real.tolist())


def test_shuffled_vec_is_deterministic_per_seed() -> None:
    real = torch.arange(16, dtype=torch.float32)
    a = shuffled_vec(real, seed=2)
    b = shuffled_vec(real, seed=2)
    c = shuffled_vec(real, seed=3)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
