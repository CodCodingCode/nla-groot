"""Tests for batched policy server helpers."""

from __future__ import annotations

import numpy as np

from nla.eval.nla_policy_server import _build_batched_options, _unbatch_action_results
from nla.eval.steerability.obs_batching import stack_nested_observations


def test_build_batched_options():
    h = 8
    opts = _build_batched_options(
        [
            {"steer_h": np.ones(h, dtype=np.float32)},
            {"steer_h": np.zeros(h, dtype=np.float32)},
        ],
        batch_size=2,
    )
    assert opts["steer_h_batch"].shape == (2, h)


def test_unbatch_action_results():
    action = {"x": np.stack([np.ones(3), np.zeros(3)], axis=0).astype(np.float32)}
    pairs = _unbatch_action_results(action, {}, batch_size=2)
    assert len(pairs) == 2
    assert np.allclose(pairs[0][0]["x"], 1.0)
    assert np.allclose(pairs[1][0]["x"], 0.0)


def test_stack_two_single_obs():
    obs = {
        "video": {"image": np.zeros((1, 1, 2, 2, 3), dtype=np.uint8)},
        "state": {"x": np.zeros((1, 1, 1), dtype=np.float32)},
        "language": {"task": [["a"]]},
    }
    batched = stack_nested_observations([obs, obs])
    assert batched["video"]["image"].shape[0] == 2
