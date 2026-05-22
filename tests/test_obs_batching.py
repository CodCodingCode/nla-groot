"""Tests for nested observation batching."""

from __future__ import annotations

import numpy as np

from nla.eval.steerability.obs_batching import (
    infer_nested_batch_size,
    stack_nested_observations,
)


def _single_obs() -> dict:
    return {
        "video": {
            "image": np.zeros((1, 1, 4, 4, 3), dtype=np.uint8),
            "wrist_image": np.zeros((1, 1, 4, 4, 3), dtype=np.uint8),
        },
        "state": {
            "x": np.zeros((1, 1, 2), dtype=np.float32),
        },
        "language": {
            "task": [["pick up the bowl"]],
        },
    }


def test_stack_nested_observations_batch_size():
    obs0 = _single_obs()
    obs1 = _single_obs()
    obs1["language"]["task"] = [["open the drawer"]]
    batched = stack_nested_observations([obs0, obs1])
    assert infer_nested_batch_size(batched) == 2
    assert batched["video"]["image"].shape[0] == 2
    assert len(batched["language"]["task"]) == 2
    assert batched["language"]["task"][0][0] == "pick up the bowl"
    assert batched["language"]["task"][1][0] == "open the drawer"
